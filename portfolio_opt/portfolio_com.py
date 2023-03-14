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

# we start by working with a 5-dimensional SDE

class PI(nn.Module):
    def __init__(self):
        super(PI, self).__init__()
        #pi_tensor = torch.abs(torch.tensor([1., 1., 1., 1., 1.], requires_grad=True)).to(device)
        self.pi = nn.Parameter(torch.abs(torch.tensor([.2, .2, .2, .2, .2], requires_grad=True)).to(device)) 
        
        return 


class FKModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.IDX_N = 0 # index for the number of realizations
        self.IDX_NX0 = 1 # index for the number of initial conditions 
        self.IDX_D = 2 # index for the number of dimensions
        self.IDX_Nt = 3 # index for the number of time steps
        
        self.N = 100 # number of realizations
        self.N_X0 = 100 # number of initial conditions
        self.D = 5 # number of dimensions
        self.Nt = 200 # number of time steps
        self.T = 2. # terminal time
        
        self.dt = self.T / self.Nt
        
        self.dW = np.sqrt(self.dt) * torch.randn(self.N, 1, self.D, self.Nt).to(device)
        self.dW1 = np.sqrt(self.dt) * torch.randn(self.N, 1, self.D, self.Nt).to(device)
        
        self.p_x0 = MultivariateNormal(5 * torch.ones(self.D).to(device), 1. * torch.eye(self.D).to(device))
        
        self.sigma = 5
        
        self.model = PI()
        
        self.metrics = torch.zeros((100,5))
        self.epochs = torch.linspace(0,99,100)

        

    def mu_em(self, x):
        x0 = x[:,:,0]
        x1 = x[:,:,1]
        x2 = x[:,:,2]
        x3 = x[:,:,3]
        x4 = x[:,:,4]
        
        mu0 = x0 + x2 + x3
        mu1 = x1 ** 2
        mu2 = torch.cos(x2)
        mu3 = torch.sin(x3)
        mu4 = x4
        mu_t = torch.cat((mu0.unsqueeze(-1), mu1.unsqueeze(-1), mu2.unsqueeze(-1), mu3.unsqueeze(-1), mu4.unsqueeze(-1)), dim=-1)
        return mu_t / 50
    
    def mu_com(self, x):
        x0 = x[:,:,0,:]
        x1 = x[:,:,1,:]
        x2 = x[:,:,2,:]
        x3 = x[:,:,3,:]
        x4 = x[:,:,4,:]
        
        mu0 = x0 + x2 + x3
        mu1 = x1 ** 2
        mu2 = torch.cos(x2)
        mu3 = torch.sin(x3)
        mu4 = x4
        mu_t = torch.cat((mu0.unsqueeze(-2), mu1.unsqueeze(-2), mu2.unsqueeze(-2), mu3.unsqueeze(-2), mu4.unsqueeze(-2)), dim=-2)
        return mu_t/50
    
    def loss(self, batch):
        X0 = batch.to(device)
        X0.requires_grad=True
        # calculating with girsanov
        
        W = self.dW.cumsum(self.IDX_Nt) * self.sigma + X0.unsqueeze(0).unsqueeze(-1).repeat(self.N, 1, 1, self.Nt)
        mu = self.mu_com(W)
        a = (mu * self.dW / self.sigma).sum(self.IDX_D,keepdims=True).sum(self.IDX_Nt)
        b = -0.5 * self.dt * (((mu / self.sigma) ** 2).sum(self.IDX_D,keepdims=True)).sum(self.IDX_Nt)
        x_end = W[:,:,:,-1]
        pi_norm = torch.norm(self.model.pi, p=1, dim=None, keepdim=False, out=None, dtype=None)
        #yT_com = torch.log((self.model.pi.unsqueeze(0).unsqueeze(0) / pi_norm * x_end).sum(-1)) + a.squeeze() + b.squeeze()
        #yT_com = (self.model.pi.unsqueeze(0).unsqueeze(0) / pi_norm * x_end).sum(-1) * (a.squeeze() + b.squeeze()).exp()
        #yT_com = x_end.sum(-1) * (a.squeeze() + b.squeeze()).exp()
        yT_com = (self.model.pi.unsqueeze(0).unsqueeze(0) / pi_norm * x_end).sum(-1) * (a.squeeze() + b.squeeze()).exp()
        #yT_com = torch.log((self.model.pi.unsqueeze(0).unsqueeze(0) / pi_norm * x_end).sum(-1)) + (a.squeeze() + b.squeeze())
        
        #X01 = X0.unsqueeze(0).repeat(self.N,1,1)
        #for idx in range(self.Nt-1):
        #    X01  = X01 + self.mu_em(X01) * self.dt + self.dW[:,:,:,idx] * self.sigma
        #pi_norm = torch.norm(self.model.pi, p=1, dim=None, keepdim=False, out=None, dtype=None)
        #x_end = X01
        #yT_em = (self.model.pi.unsqueeze(0).unsqueeze(0) / pi_norm * x_end).sum(-1)
        ##yT_em = x_end.sum(-1)
        
        return -yT_com[~torch.isnan(yT_com)].mean()#torch.abs(hamiltonian_com).mean()
        #return - yT_com.mean()




    def training_step(self, batch, batch_idx):
        X0 = torch.abs(self.p_x0.sample((self.N_X0,)))
        
        loss = self.loss(X0)
        self.log('train_loss', loss)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        X0 = torch.abs(self.p_x0.sample((self.N_X0,))).to(device)
        X0 = X0.unsqueeze(0).repeat(self.N,1,1)

        
        #X = torch.zeros_like(self.dW).repeat(1,self.N_X0,1,1).to(device)
        
        #X[0,:,:,0] = X0
        #X.requires_grad = True
        
        for idx in range(self.Nt-1):
            X0  = X0 + self.mu_em(X0) * self.dt + self.dW1[:,:,:,idx] * self.sigma
        pi_norm = torch.norm(self.model.pi, p=1, dim=None, keepdim=False, out=None, dtype=None)
        x_end = X0
        yT_com = (self.model.pi.unsqueeze(0).unsqueeze(0) / pi_norm * x_end).sum(-1)
        self.log('validation_loss', -yT_com.mean(), prog_bar=True)
        self.metrics[self.current_epoch,:] = (self.model.pi.unsqueeze(0).unsqueeze(0) / pi_norm ).squeeze()
        plt.plot(self.epochs, self.metrics[:,0], label='pi_0')
        plt.plot(self.epochs, self.metrics[:,1], label='pi_1')
        plt.plot(self.epochs, self.metrics[:,2], label='pi_2')
        plt.plot(self.epochs, self.metrics[:,3], label='pi_3')
        plt.plot(self.epochs, self.metrics[:,4], label='pi_4')
        plt.ylabel('Magnitude')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig('/scratch/xx84/meta_fk/portfolio/com.png')
        plt.clf()
        return -yT_com.mean()
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {'optimizer' : optimizer, 'scheduler': scheduler, 'monitor' : 'train_loss'}
        #return optimizers, schedulers





if __name__ == '__main__':
    pl.seed_everything(1234)
    print(sys.executable)
    device = torch.device("cuda:0")
    
    p_i = MultivariateNormal(torch.zeros(1) + torch.ones(1), torch.eye(1))
    trainset = p_i.sample([3000])
    testset = p_i.sample([300])
    train_loader = torch.utils.data.DataLoader(trainset, batch_size = 150, shuffle=True, num_workers = 1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size = 150, shuffle=True)
    model = FKModule()
    trainer = pl.Trainer(max_epochs=100,gpus=1,check_val_every_n_epoch=1)
    trainer.fit(model, train_loader, test_loader)
    
