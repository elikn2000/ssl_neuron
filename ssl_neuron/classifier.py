import torch
import numpy as np
import wandb
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ssl_neuron.graph_ops import PVManifoldMLR, predict_pvmlr
from ssl_neuron.utils import compute_eig_lapl_torch_batch
import pandas as pd
import pickle
import logging 

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

_MARA_PKL = "/user/elias.knack/u27510/ssl_neuron/ssl_neuron/data/classification/graphdino_morphological_embeddings_new.pkl"
_COREG_CSV = "/user/elias.knack/u27510/ssl_neuron/ssl_neuron/data/classification/coregistration_v654.csv"
_CELL_TYPE_PKL = "/user/elias.knack/u27510/ssl_neuron/ssl_neuron/data/classification/aibs_metamodel_mtypes_v661_v2_matver_v889.pkl"

def run_cell_type_eval(model, train_dataset, val_dataset, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), method="MLR", embedding="vector_embedding"):
    """Evaluate cell-type classification from model embeddings.

    Reuses already-loaded datasets (just switches them to inference mode temporarily)
    to avoid the slow dataset reload. Results are logged to wandb.
    """
    logging.info("Running cell-type classification eval...")

    # Switch datasets to inference mode (no augmentation, no branch dropping)
    orig_states = []
    for ds in [train_dataset, val_dataset]:
        orig_states.append((ds.inference, ds.n_drop_branch, ds.jitter_var,
                            ds.translate_var, ds.rotation_axis))
        ds.inference = True
        ds.n_drop_branch = 0
        ds.jitter_var = 0
        ds.translate_var = 0
        ds.rotation_axis = None

    # Extract embeddings, tracking which split each neuron came from
    # num_workers=0: avoid spawning worker processes after training (OOM risk)
    all_embeddings, all_cell_ids, all_splits = [], [], []
    model.eval()
    with torch.no_grad():
        for split_name, ds in [("train", train_dataset), ("val", val_dataset)]:
            loader = DataLoader(ds, batch_size=128, shuffle=False, drop_last=False,
                                num_workers=0, 
                                pin_memory=False)
            for batched_g in loader:
                f1, _, a1, _ = [x.float().to(device, non_blocking=True) for x in batched_g]
               
                # compute positional encoding
                l1 = compute_eig_lapl_torch_batch(a1)
                vector_embedding, projection = model.student_encoder(f1, a1, l1) 
                # embedding has shape (batch_size, embedding_dim); extend list with
                # per-sample embeddings so lengths match `ds.cell_ids` later.
                if embedding == "vector_embedding":
                    all_embeddings.extend(vector_embedding.cpu().numpy())
                elif embedding == "projection": 
                    all_embeddings.extend(projection.cpu().numpy())
        
            all_cell_ids.extend(ds.cell_ids)
            all_splits.extend([split_name] * ds.num_samples)
    # Restore dataset state
    for ds, state in zip([train_dataset, val_dataset], orig_states):
        ds.inference, ds.n_drop_branch, ds.jitter_var, ds.translate_var, ds.rotation_axis = state

    seg2emb = {int(cid.split("_")[0]): emb for cid, emb in zip(all_cell_ids, all_embeddings)}
    seg2split = {int(cid.split("_")[0]): s for cid, s in zip(all_cell_ids, all_splits)}

    # Load label data
    with open(_MARA_PKL, "rb") as f:
        mara_raw = pickle.load(f)
    morpho_df = pd.DataFrame.from_dict(mara_raw).drop_duplicates(subset="nucleus_id", keep="first")
    nuc2seg = dict(zip(morpho_df["nucleus_id"], morpho_df["segment_id"]))

    with open(_CELL_TYPE_PKL, "rb") as f:
        cell_type_pd = pickle.load(f)
    cell_types = cell_type_pd[["nucleus_id", "cell_type"]]
    num_total_types = cell_type_pd["cell_type"].nunique()
    coreg = pd.read_csv(_COREG_CSV, index_col=0)
    if "cell_type" in coreg.columns:
        coreg = coreg.drop(columns=["cell_type"])
    df = coreg.merge(cell_types, on="nucleus_id", how="left")

    def lookup(nid):
        seg_id = nuc2seg.get(nid)
        return seg2emb.get(int(seg_id)) if seg_id is not None else None

    df["emb"] = df["nucleus_id"].apply(lookup)
    df["gmae_split"] = df["nucleus_id"].apply(lambda nid: seg2split.get(int(nuc2seg.get(nid, -1)), None) if nuc2seg.get(nid) is not None else None)
    df = df.dropna(subset=["emb", "cell_type"]).reset_index(drop=True)
    logging.info(f"Cell-type eval: {len(df)} neurons with embeddings + labels "
                 f"({(df['gmae_split']=='train').sum()} gmae-train, {(df['gmae_split']=='val').sum()} gmae-val)")

    X = np.stack(df["emb"].values)
    y = df["cell_type"].values
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    if method == "MLR":
        # multi_class='multinomial' uses the cross-entropy loss (softmax)
        # solver='lbfgs' is the default and supports multinomial regression
        MLR = LogisticRegression(multi_class='multinomial', solver='lbfgs')

        # Training
        MLR.fit(x_train, y_train)

        # Predictions
        MLR_acc = accuracy_score(y_test, MLR.predict(x_test)) 
        MLR_bal = balanced_accuracy_score(y_test, MLR.predict(x_test))
        logging.info(f"MLRCV      | acc={MLR_acc:.4f}  bal_acc={MLR_bal:.4f}")
        logging.info(f"GraphDINO ref| acc=0.4411  bal_acc=0.3413")

        # Overfitting check: fit Ridge on gmae-train neurons, test on gmae-val neurons
        df_tr = df[df["gmae_split"] == "train"].reset_index(drop=True)
        df_vl = df[df["gmae_split"] == "val"].reset_index(drop=True)
        if len(df_tr) > 10 and len(df_vl) > 10:
            X_tr = np.stack(df_tr["emb"].values)
            y_tr = df_tr["cell_type"].values
            X_vl = np.stack(df_vl["emb"].values)
            y_vl = df_vl["cell_type"].values
            MLR_overfit = LogisticRegression(multi_class='multinomial', solver='lbfgs')
            MLR_overfit.fit(X_tr, y_tr)
            MLR_overfit_train_acc = accuracy_score(y_tr, MLR_overfit.predict(X_tr))
            MLR_overfit_train_bal = balanced_accuracy_score(y_tr, MLR_overfit.predict(X_tr))
            MLR_overfit_val_acc = accuracy_score(y_vl, MLR_overfit.predict(X_vl))
            MLR_overfit_val_bal = balanced_accuracy_score(y_vl, MLR_overfit.predict(X_vl))
            logging.info(f"Overfit check (MLR fit on gmae-train, eval on gmae-val):")
            logging.info(f"  gmae-train acc={MLR_overfit_train_acc:.4f}  bal_acc={MLR_overfit_train_bal:.4f} (n={len(df_tr)})")
            logging.info(f"  gmae-val   acc={MLR_overfit_val_acc:.4f}  bal_acc={MLR_overfit_val_bal:.4f} (n={len(df_vl)})")
            wandb.log({
                "eval/overfit_MLR_train_acc": MLR_overfit_train_acc,
                "eval/overfit_MLR_train_bal": MLR_overfit_train_bal,
                "eval/overfit_MLR_val_acc": MLR_overfit_val_acc,
                "eval/overfit_MLR_val_bal": MLR_overfit_val_bal,
            })

        wandb.log({
            "eval/MLR_acc": MLR_acc, "eval/MLR_bal_acc": MLR_bal,
        })
        acc=MLR_acc
        bal=MLR_bal
        
    if method == "Hyperbolic_MLR":
        
        HYP_MLR =train_hyperbolic_mlr(x_train, y_train, num_classes=num_total_types, k=-1.0, epochs=100, lr=0.01, batch_size=128, device=device)

     
        # Predictions
        HYP_MLR_acc = accuracy_score(y_test, HYP_MLR.predict(x_test)) 
        HYP_MLR_bal = balanced_accuracy_score(y_test, HYP_MLR.predict(x_test))

        logging.info(f"HYP_MLRCV      | acc={HYP_MLR_acc:.4f}  bal_acc={HYP_MLR_bal:.4f}")
        logging.info(f"GraphDINO ref| acc=0.4411  bal_acc=0.3413")

        # Overfitting check: fit Ridge on gmae-train neurons, test on gmae-val neurons
        df_tr = df[df["gmae_split"] == "train"].reset_index(drop=True)
        df_vl = df[df["gmae_split"] == "val"].reset_index(drop=True)
        if len(df_tr) > 10 and len(df_vl) > 10:
            X_tr = np.stack(df_tr["emb"].values)
            y_tr = df_tr["cell_type"].values
            X_vl = np.stack(df_vl["emb"].values)
            y_vl = df_vl["cell_type"].values
            HYP_MLR_overfit = LogisticRegression(multi_class='multinomial', solver='lbfgs')
            HYP_MLR_overfit.fit(X_tr, y_tr)
            HYP_MLR_overfit_train_acc = accuracy_score(y_tr, HYP_MLR_overfit.predict(X_tr))
            HYP_MLR_overfit_train_bal = balanced_accuracy_score(y_tr, HYP_MLR_overfit.predict(X_tr))
            HYP_MLR_overfit_val_acc = accuracy_score(y_vl, HYP_MLR_overfit.predict(X_vl))
            HYP_MLR_overfit_val_bal = balanced_accuracy_score(y_vl, HYP_MLR_overfit.predict(X_vl))
            logging.info(f"Overfit check (HYP_MLR fit on gmae-train, eval on gmae-val):")
            logging.info(f"  gmae-train acc={HYP_MLR_overfit_train_acc:.4f}  bal_acc={HYP_MLR_overfit_train_bal:.4f} (n={len(df_tr)})")
            logging.info(f"  gmae-val   acc={HYP_MLR_overfit_val_acc:.4f}  bal_acc={HYP_MLR_overfit_val_bal:.4f} (n={len(df_vl)})")
            wandb.log({
                "eval/overfit_HYP_MLR_train_acc": HYP_MLR_overfit_train_acc,
                "eval/overfit_HYP_MLR_train_bal": HYP_MLR_overfit_train_bal,
                "eval/overfit_HYP_MLR_val_acc": HYP_MLR_overfit_val_acc,
                "eval/overfit_HYP_MLR_val_bal": HYP_MLR_overfit_val_bal,
            })

        wandb.log({
            "eval/HYP_MLR_acc": HYP_MLR_acc, "eval/HYP_MLR_bal_acc": HYP_MLR_bal,
        })
        acc=HYP_MLR_acc
        bal=HYP_MLR_bal

    if method == "ridge":
        clf_ridge = RidgeClassifierCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
        clf_ridge.fit(x_train, y_train)
        ridge_acc = accuracy_score(y_test, clf_ridge.predict(x_test))
        ridge_bal = balanced_accuracy_score(y_test, clf_ridge.predict(x_test))

        

        logging.info(f"RidgeCV      | acc={ridge_acc:.4f}  bal_acc={ridge_bal:.4f}")
        logging.info(f"GraphDINO ref| acc=0.4411  bal_acc=0.3413")

        # Overfitting check: fit Ridge on gmae-train neurons, test on gmae-val neurons
        df_tr = df[df["gmae_split"] == "train"].reset_index(drop=True)
        df_vl = df[df["gmae_split"] == "val"].reset_index(drop=True)
        if len(df_tr) > 10 and len(df_vl) > 10:
            X_tr = np.stack(df_tr["emb"].values)
            y_tr = df_tr["cell_type"].values
            X_vl = np.stack(df_vl["emb"].values)
            y_vl = df_vl["cell_type"].values
            clf_overfit = RidgeClassifierCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
            clf_overfit.fit(X_tr, y_tr)
            ridge_overfit_train_acc = accuracy_score(y_tr, clf_overfit.predict(X_tr))
            ridge_overfit_train_bal = balanced_accuracy_score(y_tr, clf_overfit.predict(X_tr))
            ridge_overfit_val_acc = accuracy_score(y_vl, clf_overfit.predict(X_vl))
            ridge_overfit_val_bal = balanced_accuracy_score(y_vl, clf_overfit.predict(X_vl))
            logging.info(f"Overfit check (Ridge fit on gmae-train, eval on gmae-val):")
            logging.info(f"  gmae-train acc={ridge_overfit_train_acc:.4f}  bal_acc={ridge_overfit_train_bal:.4f} (n={len(df_tr)})")
            logging.info(f"  gmae-val   acc={ridge_overfit_val_acc:.4f}  bal_acc={ridge_overfit_val_bal:.4f} (n={len(df_vl)})")
            wandb.log({
                "eval/overfit_ridge_train_acc": ridge_overfit_train_acc,
                "eval/overfit_ridge_train_bal": ridge_overfit_train_bal,
                "eval/overfit_ridge_val_acc": ridge_overfit_val_acc,
                "eval/overfit_ridge_val_bal": ridge_overfit_val_bal,
            })

        wandb.log({
            "eval/ridge_acc": ridge_acc, "eval/ridge_bal_acc": ridge_bal,
        })
        acc=ridge_acc
        bal=ridge_bal
    return  acc, bal





def train_hyperbolic_mlr(X, y, num_classes, k=-1.0, epochs=100, lr=0.01, batch_size=64, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    """
    Trains the PVManifoldMLR using precalculated embeddings.
    Accepts string labels by factorizing them and returns a wrapper with a
    sklearn-like `predict` that returns labels in the original form.

    Args:
        X: np.array or torch.Tensor of shape [N, embedding_dim]
        y: np.array or torch.Tensor of shape [N] (integer or string labels)
        num_classes: Total unique cell types (if None, inferred from y)
        k: Curvature (must be negative)
    """
    # 1. Prepare Data
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)

    # Work on a numpy view of y to detect string/object dtypes
    if isinstance(y, torch.Tensor):
        y_arr = y.cpu().numpy()
    else:
        y_arr = np.array(y)

    classes = None
    # Detect string/object labels and factorize
    if y_arr.dtype.kind in {"U", "S", "O"}:
        unique_labels, inv = np.unique(y_arr, return_inverse=True)
        y_int = inv.astype(np.int64)
        classes = unique_labels
        detected_num = len(unique_labels)
        if num_classes is None:
            num_classes = detected_num
        elif num_classes != detected_num:
            logging.warning(f"num_classes ({num_classes}) != detected unique labels ({detected_num}); using detected value")
            num_classes = detected_num
    else:
        # Assume numeric labels (integers)
        y_int = y_arr.astype(np.int64)

    y_tensor = torch.tensor(y_int, dtype=torch.long)
    dataset = TensorDataset(X, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Initialize Layer
    input_dim = X.shape[1]
    mlr_layer = PVManifoldMLR(k=k, in_features=input_dim, num_classes=num_classes).to(device)
    
    optimizer = optim.Adam(mlr_layer.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 3. Training Loop
    mlr_layer.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = mlr_layer(batch_x)
            loss = criterion(logits, batch_y)
            
            loss.backward()
            # Gradient clipping is highly recommended for hyperbolic layers
            torch.nn.utils.clip_grad_norm_(mlr_layer.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = logits.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc = 100. * correct / total
            print(f"Epoch {epoch+1:3d}: Loss = {epoch_loss/len(loader):.4f}, Acc = {acc:.2f}%")
            
    # Wrap model to provide a sklearn-like `predict` that returns labels in the
    # same form as the provided `y` (strings or integers).
    class _PVMLRWrapper:
        def __init__(self, model, classes):
            self.model = model
            self._classes = classes

        def predict(self, X_in):
            # Get integer predictions first
            idxs = predict_pvmlr(self.model, X_in)
            idxs_np = idxs.cpu().numpy()
            if self._classes is None:
                return idxs_np
            return np.asarray(self._classes)[idxs_np]

        # expose the underlying pytorch model if needed
        @property
        def model_(self):
            return self.model

    return _PVMLRWrapper(mlr_layer, classes)