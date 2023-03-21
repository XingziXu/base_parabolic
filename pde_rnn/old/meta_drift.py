import torch
from torch import nn
from torch.autograd import grad
from torch.nn import functional as F
import os
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
from geomloss import SamplesLoss
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
import time
from torchvision.utils import make_grid
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.datasets import make_moons
import sys
import math
from sklearn.datasets import make_moons

class SMLP(nn.Module):
    def __init__(self, input_size, hidden_size, layers, out_size, act=nn.Tanh()):
        super(SMLP, self).__init__()

        self.act = act

        self.fc1 = nn.Linear(input_size,hidden_size)
        mid_list = []
        for i in range(layers):
           mid_list += [nn.Linear(hidden_size,hidden_size), act]

        self.mid = nn.Sequential(*mid_list)
        self.out = nn.Linear(hidden_size, out_size, bias=False)

    def forward(self,x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.mid(out)
        out = self.out(out)
        return out    


class FKModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.num_coef = 50
        self.num_x = 50
        self.num_t = 50
        
        self.C = .3
        self.X = 1.
        self.T = 0.01
        
        self.model = SMLP(input_size = 2, hidden_size = 64, layers = 3, out_size = 1)
        self.epochs = torch.linspace(0,49,50)
        self.losses = torch.zeros(50)
        
        self.coef = torch.rand(self.num_coef, 6).to(device) * self.C
        self.x = torch.rand(self.num_x, 1).to(device) * self.X
        self.t = torch.rand(self.num_t, 1).to(device) * self.T

    def mu(self, coef, x, t):
        x0 = x ** 0
        x1 = x ** 1
        x2 = x ** 2
        x3 = x ** 3
        t1 = t ** 1
        t2 = t ** 2
        return (torch.cat((x0,x1,x2,x3,t1,t2),dim=-1) * coef).sum(-1)
    
    def loss(self, coef, x, t):
        truth = self.mu(coef, x, t)
        pred = self.model(torch.cat((x,t),dim=-1))

        return ((pred.squeeze()-truth) ** 2).mean()

    def training_step(self, batch, batch_idx):
        #coef = torch.rand(self.num_coef, 6).to(device)
        #x = torch.rand(self.num_x, 1).to(device) * self.X
        #t = torch.rand(self.num_t, 1).to(device) * self.T
        loss = self.loss(self.coef, self.x, self.t)
        self.log('train_loss', loss)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        torch.save(self.model.state_dict(), '/scratch/xx84/girsanov/pde_rnn/drift_prior.pt')
        self.losses[self.current_epoch] = self.loss(self.coef, self.x, self.t)
        plt.plot(self.epochs, self.losses)
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig('/scratch/xx84/girsanov/pde_rnn/loss.png')
        plt.clf()
        return 
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {'optimizer' : optimizer, 'scheduler': scheduler, 'monitor' : 'train_loss'}





if __name__ == '__main__':
    pl.seed_everything(1234)
    print(sys.executable)
    device = torch.device("cuda:0")
    
    p_i = MultivariateNormal(torch.zeros(1) + torch.ones(1), torch.eye(1))
    trainset = p_i.sample([3000])
    testset = p_i.sample([300])
    train_loader = torch.utils.data.DataLoader(trainset, batch_size = 30, shuffle=True, num_workers = 1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size = 30, shuffle=True)
    model = FKModule()
    trainer = pl.Trainer(max_epochs=50,gpus=1,check_val_every_n_epoch=1)
    trainer.fit(model, train_loader, test_loader)
    
