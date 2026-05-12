import os
import torch
import wandb
import torch.optim as optim
from ssl_neuron.utils import AverageMeter, compute_eig_lapl_torch_batch,  neighbors_to_adjacency_torch
class Trainer(object):
    def __init__(self, config, model, dataloaders):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.config = config
        self.model_name = config['model']['name']
        self.ckpt_dir = config['trainer']['ckpt_dir']
        self.save_every = config['trainer']['save_ckpt_every']
        self.plot_latents_every = config['trainer']['plot_latents_every']
        ### datasets
        self.train_loader = dataloaders[0]
        self.val_loader= dataloaders[1]

        ### trainings params
        self.max_iter = config['optimizer']['max_iter']
        self.init_lr = config['optimizer']['lr']
        self.exp_decay = config['optimizer']['exp_decay']
        self.lr_warmup = torch.linspace(0., self.init_lr,  steps=(self.max_iter // 50)+1)[1:]
        self.lr_decay = self.max_iter // 5
        
        self.optimizer = optim.Adam(list(self.model.parameters()), lr=0)
        
      
    def set_lr(self): 
        if self.curr_iter < len(self.lr_warmup):
            lr = self.lr_warmup[self.curr_iter]
        else:
            lr = self.init_lr * self.exp_decay ** ((self.curr_iter - len(self.lr_warmup)) / self.lr_decay)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr
        

    def train(self):     
        self.curr_iter = 0
        epoch = 0
        while self.curr_iter < self.max_iter:
            # Run one epoch.
            self._train_epoch(epoch)

            if epoch % self.save_every == 0:
                # Save checkpoint.
                self._save_checkpoint(epoch)
           

            epoch += 1


    def _train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        max_val = 0
        teacher_logits= AverageMeter()
        for i, data in enumerate(self.train_loader, 0):
            f1, f2, a1, a2 = [x.float().to(self.device, non_blocking=True) for x in data]
            n = a1.shape[0]

            # compute positional encoding
            l1 = compute_eig_lapl_torch_batch(a1)
            l2 = compute_eig_lapl_torch_batch(a2)
            
            self.lr = self.set_lr()
            self.optimizer.zero_grad(set_to_none=True)
            
            loss, max_val_batch, teacher_logits_avg_avg = self.model(f1, f2, a1, a2, l1, l2)
            max_val= max(max_val, max_val_batch)    

            # optimize 
            loss.sum().backward()
            self.optimizer.step()
            
            # update teacher weights
            self.model.update_moving_average()
            
            losses.update(loss.detach(), n)
            teacher_logits.update(teacher_logits_avg_avg.detach(), n)
            self.curr_iter += 1

        #print('Epoch {} | Loss {:.4f}'.format(epoch, losses.avg))
        wandb.log({'loss_train': losses.avg})
        wandb.log({'teacher_logits_avg': torch.norm(teacher_logits.avg, dim=-1)})
        wandb.log({'max_val': max_val})
    def plot_latents(self):
        self.model.eval()
        dset= self.val_loader.dataset   

        latents = np.zeros((dset.num_samples, config['model']['dim']))

        learning_rate = 5.0
        learning_rate_for_h_loss = 0.1
        perplexity = 20
        early_exaggeration = 1.0
        student_t_gamma = 0.1

        for i in tqdm(range(dset.num_samples)):
            feat, neigh = dset.__getsingleitem__(i)
            adj = neighbors_to_adjacency_torch(neigh, list(neigh.keys())).float().to(device)[None, ]
            lapl = compute_eig_lapl_torch_batch(adj, pos_enc_dim=config['model']['pos_dim']).float().to(device)
            feat = torch.from_numpy(feat).float().to(device)[None, ]
    
            latents[i] = model.student_encoder.forward(feat, adj, lapl)[0].cpu().detach()

        Poincare_Latents=PV_to_Poincare(latents,-1)

        tsne_embeddings, HT_SNE_embeddings, CO_SNE_embedding  = run_TSNE(Poincare_Latents, learning_rate, learning_rate_for_h_loss, perplexity, early_exaggeration, student_t_gamma)
        plot_low_dims(tsne_embeddings, HT_SNE_embeddings, CO_SNE_embedding, colors, learning_rate, learning_rate_for_h_loss, perplexity, early_exaggeration, student_t_gamma) 
        
    def _save_checkpoint(self, epoch):
        filename = '{}_{}.pt'.format(self.model_name, epoch)
        PATH = os.path.join(self.ckpt_dir, filename)
        torch.save(self.model.state_dict(), PATH)
        print('Save model after epoch {} as {}.'.format(epoch, filename))