from logging import config
import os
import torch
import wandb
import torch.optim as optim
from ssl_neuron.utils import AverageMeter, compute_eig_lapl_torch_batch,  neighbors_to_adjacency_torch
from ssl_neuron.classifier import run_cell_type_eval
class Trainer(object):
    def __init__(self, config, model, dataloaders):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.config = config
        self.model_name = config['model']['name']
        self.ckpt_dir = config['trainer']['ckpt_dir']
        self.save_every = config['trainer']['save_ckpt_every']
        self.plot_latents_every = config['trainer']['plot_latents_every']
        self.classification_every = config['trainer']['classification_every']
        ### datasets
        self.train_loader = dataloaders[0]
        self.val_loader= dataloaders[1]
        self.loc_embedding=config['trainer']['loc_embedding']
        ### trainings params
        self.max_iter = config['optimizer']['max_iter']
        self.init_lr = config['optimizer']['lr']
        self.exp_decay = config['optimizer']['exp_decay']
        self.lr_warmup = torch.linspace(0., self.init_lr,  steps=(self.max_iter // 50)+1)[1:]
        self.lr_decay = self.max_iter // 5

        self.optimizer = optim.Adam(list(self.model.parameters()), lr=self.init_lr)
        self.use_plateau_lr = config['optimizer']['lr_scheduler_enabled']
        if self.use_plateau_lr:
            plateau_cfg = config.get('optimizer', {}).get('lr_scheduler', {})
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=plateau_cfg.get('mode', 'min'),
                factor=plateau_cfg.get('factor', 0.5),
                patience=plateau_cfg.get('patience', 5),
                threshold=plateau_cfg.get('threshold', 1e-4),
                cooldown=plateau_cfg.get('cooldown', 0),
                min_lr=plateau_cfg.get('min_lr', 1e-6),
                verbose=plateau_cfg.get('verbose', False),
            )
        else:
            self.lr_scheduler = None
        
      
    def set_lr(self): 
        if self.curr_iter < len(self.lr_warmup):
            lr = self.lr_warmup[self.curr_iter]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr

        if self.use_plateau_lr:
            return self.optimizer.param_groups[0]['lr']

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
        self._save_checkpoint(epoch)

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
            loss, max_val_batch, teacher_logits_avg_avg = self.model(f1, f2, a1, a2, l1, l2, self.loc_embedding)
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
        wandb.log({'teacher_logits_avgnorm': torch.norm(teacher_logits.avg, dim=-1)})
        wandb.log({'max_norm': max_val})
        wandb.log({'lr': self.lr})
        wandb.log({'epoch': epoch})
        self.model.eval()
        val_losses = AverageMeter()
        with torch.no_grad():
            for data in self.val_loader:
                f1, f2, a1, a2 = [x.float().to(self.device, non_blocking=True) for x in data]
                n = a1.shape[0]

                l1 = compute_eig_lapl_torch_batch(a1)
                l2 = compute_eig_lapl_torch_batch(a2)

                loss, _, _ = self.model(f1, f2, a1, a2, l1, l2, self.loc_embedding)
                val_losses.update(loss.detach(), n)

        wandb.log({'loss_val': val_losses.avg})
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(val_losses.avg)
        wandb.log({'lr': self.optimizer.param_groups[0]['lr']})

        if (self.classification_every > 0 and epoch % self.classification_every == 0):
            run_cell_type_eval(self.model, self.train_loader.dataset, self.val_loader.dataset, self.model_name, self.device,  method=self.config['testing']['Classifier'], num_hidden=self.config['testing']['num_hidden'], dim_hidden=self.config['testing']['dim_hidden'], embedding=self.config['testing']['embedding'], save_embedding_label=False, to_wandb=True, save_model=False, save_confusion_matrix=False, loc_embedding=0, k=self.config['model']['curvature'])
        elif self.curr_iter == self.max_iter - 1:
            run_cell_type_eval(self.model, self.train_loader.dataset, self.val_loader.dataset, self.model_name, self.device,  method=self.config['testing']['Classifier'], num_hidden=self.config['testing']['num_hidden'], dim_hidden=self.config['testing']['dim_hidden'], embedding=self.config['testing']['embedding'], save_embedding_label=self.config['testing']['save_embedding_label'], to_wandb=self.config['testing']['to_wandb'], save_model=self.config['testing']['save_model'], save_confusion_matrix=self.config['testing']['save_confusion_matrix'], loc_embedding=0, k=self.config['model']['curvature'])
        self.model.train()

    def _save_checkpoint(self, epoch):
        filename = '{}_{}.pt'.format(self.model_name, epoch)
        PATH = os.path.join(self.ckpt_dir, filename)
        torch.save(self.model.state_dict(), PATH)
        print('Save model after epoch {} as {}.'.format(epoch, filename))