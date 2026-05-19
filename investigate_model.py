import json
import torch
from tqdm import tqdm
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
from ssl_neuron.datasets import GraphDataset
from ssl_neuron.utils import plot_neuron, plot_tsne, neighbors_to_adjacency_torch, compute_eig_lapl_torch_batch, PV_to_Poincare
from ssl_neuron.graphdino import create_model
from ssl_neuron.datasets import build_dataloader
from ssl_neuron.classifier import run_cell_type_eval
import wandb


config = json.load(open('ssl_neuron/configs/config.json'))

model = create_model(config)
model_name = config['model']['name']
classifier=config['testing']['Classifier']
epoch = config['testing']['epoch']



state_dict = torch.load('ssl_neuron/ckpts/{}_{}.pt'.format(model_name, epoch), map_location=torch.device('cpu'))

model.load_state_dict(state_dict)

model.eval()
if torch.cuda.is_available():
    model.cuda()
 


def get_latents(model):
    dset = GraphDataset(config, mode='test', max_samples=1000)
    latents_dir = "/user/elias.knack/u27510/ssl_neuron/ssl_neuron/data_latents/latents_{}_{}.pt".format(model_name, epoch) 
    latents = torch.zeros((dset.num_samples, config['model']['dim']))

    for i in tqdm(range(dset.num_samples)):
        feat, neigh = dset.__getsingleitem__(i)
        adj = neighbors_to_adjacency_torch(neigh, list(neigh.keys())).float().cuda()[None, ]
        lapl = compute_eig_lapl_torch_batch(adj, pos_enc_dim=config['model']['pos_dim']).float().cuda()
        feat = torch.from_numpy(feat).float().cuda()[None, ]
        
        latents[i] = model.student_encoder.forward(feat, adj, lapl)[0].cpu().detach()
        
    latents=PV_to_Poincare(latents,K=-1.0)
    torch.save(latents, latents_dir)
def classify(model):
    wandb.login()
    wandb.init(project='SSL_Neuron_Hyperbolic_embedding', entity="ecker-lab", config=config, name="Classifier_{}_{}_{}".format(classifier, model_name, epoch))
    train_dataset = GraphDataset(config, mode='train')
    val_dataset = GraphDataset(config, mode='val')
    run_cell_type_eval(model, train_dataset, val_dataset, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), method=classifier, embedding=config['testing']['embedding'])

classify(model)