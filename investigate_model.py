import json
import torch
from tqdm import tqdm
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
from ssl_neuron.datasets import GraphDataset
from ssl_neuron.utils import plot_neuron, plot_tsne, neighbors_to_adjacency_torch, compute_eig_lapl_torch_batch, PV_to_Poincare
from ssl_neuron.graphdino import create_model

config = json.load(open('ssl_neuron/configs/config.json'))

model = create_model(config)
model_name = config['model']['name']
epoch = config['testing']['epoch']
latents_dir = "/user/elias.knack/u27510/ssl_neuron/ssl_neuron/data_latents/latents_{}_{}.pt".format(model_name, epoch)  
state_dict = torch.load('ssl_neuron/ckpts/{}_{}.pt'.format(model_name, epoch))

model.load_state_dict(state_dict)

model.eval()
model.cuda()

dset = GraphDataset(config, mode='test', max_samples=1000)

latents = torch.zeros((dset.num_samples, config['model']['dim']))

for i in tqdm(range(dset.num_samples)):
    feat, neigh = dset.__getsingleitem__(i)
    adj = neighbors_to_adjacency_torch(neigh, list(neigh.keys())).float().cuda()[None, ]
    lapl = compute_eig_lapl_torch_batch(adj, pos_enc_dim=config['model']['pos_dim']).float().cuda()
    feat = torch.from_numpy(feat).float().cuda()[None, ]
    
    latents[i] = model.student_encoder.forward(feat, adj, lapl)[0].cpu().detach()
    
latents=PV_to_Poincare(latents,K=-1.0)
torch.save(latents, latents_dir)