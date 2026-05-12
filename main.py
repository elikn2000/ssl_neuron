import json
import argparse
import wandb
from ssl_neuron.train import Trainer
from ssl_neuron.graphdino import create_model
from ssl_neuron.datasets import build_dataloader

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to config file.', type=str, default='./ssl_neuron/configs/config.json')


def main(args):
    # load config
    config = json.load(open(args.config))
    
    # load data
    print('Loading dataset: {}'.format(config['data']['class']))
    train_loader, val_loader = build_dataloader(config)

    # build model 
    model = create_model(config)
    wandb.login()
    wandb.init(project='SSL_Neuron_Hyperbolic_embedding', entity="ecker-lab", config=config, name=config['model']['name'])
    trainer = Trainer(config, model, [train_loader, val_loader])

    print('Start training.')
    trainer.train()
    print('Done.')


if __name__ == '__main__':    
    args = parser.parse_args()
    main(args)