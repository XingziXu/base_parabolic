import torch
from torch import nn
from torch.autograd import grad
from torch.nn import functional as F
import os
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np

from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader

from torchvision.utils import make_grid
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.datasets import make_moons
import sys
from random import randint
import seaborn as sns 
import pandas as pd
from neuralop.models import TFNO



def drift(x, coef):
    x = x.unsqueeze(-1)
    x0 = x ** 0
    x1 = x ** 1
    x2 = x ** 2
    vals = torch.cat((x0,x1,x2),axis=-1)
    return (coef * vals).sum(-1)

def diffusion(x,t):
    return 1
def initial(x):
    return torch.sin(6*np.pi*x)
def initial_val(x):
    return torch.sin(1*np.pi*x)
def r_value():
    return 1


class FKModule(pl.LightningModule):
    def __init__(self, lr = 1e-3, dim = 2, batch_size = 100, res_spa=50):
        super().__init__()
        
        
        self.metrics = torch.zeros((50,1))
        self.epochs = torch.linspace(0,49,50)
        operator = TFNO(n_modes=(16, 16), hidden_channels=64,
                in_channels=3, out_channels=1)
        self.fno = TFNO(n_modes=(16, 16), in_channels=1, hidden_channels=32, out_channels=1, projection_channels=64, factorization='tucker', rank=0.42)

    def loss(self, xt, coef):

        return 

    def training_step(self, batch, batch_idx):
        # REQUIRED
        xt = batch.to(device)
        u_em, u_gir, u_rnn = self.loss(xt, coef=torch.rand(1,1,1,3).to(device))
        loss = torch.norm((u_rnn-u_em))/torch.norm(u_gir)
        #tensorboard_logs = {'train_loss': loss_prior}
        self.log('train_loss', loss)
        #print(loss_total)
        return {'loss': loss}
        
        
    def validation_step(self, batch, batch_idx):

        return #{'loss': loss_total}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.sequence.gru.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}



if __name__ == '__main__':
    pl.seed_everything(1234)
    print(sys.executable)
    #dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    #mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
    #mnist_train, mnist_val = random_split(dataset, [55000,5000])
    device = torch.device("cpu")
    
    X = 0.5
    T = 0.2
    #num_time = 50
    dim = 10
    res_spa = 16
    res_time = 60
    batch_size = 25
    N = 1000
    xs = torch.linspace(0., 1., res_spa).unsqueeze(1).repeat(1,dim) * X
    ts = torch.linspace(0., 1., res_time).unsqueeze(1) * T
    dt = ts[0,0]-ts[1,0]
    coef=torch.rand(1,1,1,3).to(device)
    dB = np.sqrt(dt.item()) * np.random.randn(ts.shape[0], N, res_spa, dim)
    dB[0,:,:,:] = 0 
    B0 = dB.copy()
    B0 = torch.Tensor(B0.cumsum(0)).to(device)
    dB = torch.Tensor(dB).to(device)
    Bx = (xs.unsqueeze(0).unsqueeze(0)+B0)
    p0Bx = initial(Bx)
    muBx = drift(Bx, coef)
    expmart = torch.exp(torch.cumsum(muBx*dB,dim=0) - 0.5 * torch.cumsum((muBx ** 2) * dt,dim=0))
    u_gir = (p0Bx*expmart).mean(1)
    
    #dataset = torch.cat((xs,ts),dim=0)
    data_train = u_gir[:50,:,:,:]
    data_val = u_gir[51:,:,:,:]
    
    train_kwargs = {'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 1}

    test_kwargs = {'batch_size': batch_size,
            'shuffle': False,
            'num_workers': 1}

    train_loader = torch.utils.data.DataLoader(data_train,**train_kwargs)
    val_loader = torch.utils.data.DataLoader(data_val, **test_kwargs)

    model = FKModule(batch_size=batch_size, dim=dim, res_spa=res_spa)
    trainer = pl.Trainer(max_epochs=50,gpus=1)
    trainer.fit(model, train_loader, val_loader)
    
    print(trainer.logged_metrics['val_loss'])
    print(trainer.logged_metrics['train_loss'])