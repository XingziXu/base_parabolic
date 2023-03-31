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
#import deepxde as dde
#import matplotlib.pyplot as plt
#import numpy as np

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
    return torch.sin(6*np.pi*x).sum(-1)
def initial_val(x):
    return torch.sin(1*np.pi*x)
def r_value():
    return 1

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x): 
        return torch.sigmoid(x)*x

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.act = Swish()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.hidden_fc = nn.Linear(hidden_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h_1 = torch.tanh(self.input_fc(x))
        h_2 = torch.tanh(self.hidden_fc(h_1))
        y_pred = self.output_fc(h_2)
        return y_pred

class FKModule(pl.LightningModule):
    def __init__(self, m=100, dim=10, p=15, batch_size = 100, lr=1e-3, X=1.0, T=1.0, N=1000, num_time=50):
        super().__init__()
        self.num_time = num_time
        self.N = N # number of instances we estimate our PDE solutions with
        self.X = X # interval size
        self.T = T # time interval size
        self.m = m # number of "sensors" for the function u
        self.dim = dim # number of dimension of x
        self.p = p # number of "branches"
        self.lr = lr # learning rate
        self.sensors = initial(torch.linspace(0., 1., self.m).unsqueeze(-1).repeat(1,self.dim) * self.X)
        self.batch_size = batch_size
        
        self.branch = MLP(input_dim=self.m, hidden_dim=100, output_dim=self.p) # branch network
        self.trunk = MLP(input_dim=dim+1, hidden_dim=16, output_dim=self.p) # trunk network
        
        self.t = torch.linspace(0,1,steps=self.num_time)* self.T
        self.dt = self.t[1]-self.t[0] # define time step
        # define the brwonian motion starting at zero
        self.dB = np.sqrt(self.dt.item()) * np.random.randn(self.t.shape[0], self.N, self.batch_size, self.dim)
        self.dB[0,:,:,:] = 0 
        self.B0 = self.dB.copy()
        self.B0 = torch.Tensor(self.B0.cumsum(0)).to(device)
        self.dB = torch.Tensor(self.dB).to(device)
        
        self.metrics = torch.zeros((50,1))
        self.epochs = torch.linspace(0,49,50)

    def loss(self, xt, coef):
        xs = xt[:,:-1]
        ts = xt[:,-1]
        coef = coef
        Bx = (xs.unsqueeze(0).unsqueeze(0)+self.B0)
        p0Bx = initial(Bx)
        muBx = drift(Bx, coef)
        expmart = torch.exp(torch.cumsum((muBx*self.dB).sum(-1),dim=0) - 0.5 * torch.cumsum(((muBx ** 2).sum(-1)) * self.dt,dim=0))
        u_gir = (p0Bx*expmart).mean(1)
        branchs = self.branch(self.sensors.unsqueeze(0)).repeat(self.batch_size, 1)
        pred = torch.zeros_like(u_gir)

        u_gir_selected = torch.zeros(self.batch_size)
        for i in (0,self.batch_size-1):
            u_gir_selected[i] = u_gir[int(ts[i]),i]
        trunks = self.trunk(torch.cat((xs,ts.unsqueeze(1)*self.dt),dim=1))
        pred = (branchs * trunks).sum(1)
        return torch.norm(pred-u_gir_selected, p=1)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        xt = batch.to(device)
        loss = self.loss(xt, coef=torch.Tensor([[[[0.2322, 0.8852, 0.5074]]]]))
        self.log('train_loss', loss)
        return {'loss': loss}
        
    def validation_step(self, batch, batch_idx):
        return #{'loss': loss_total}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}



if __name__ == '__main__':
    pl.seed_everything(1234)
    print(sys.executable)
    #dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    #mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
    #mnist_train, mnist_val = random_split(dataset, [55000,5000])
    device = torch.device("cpu")
    
    m=100
    p=15
    X = 0.5
    T = 0.01
    num_time = 50
    dim = 10
    num_samples = 20025
    batch_size = 25
    N = 1000
    xs = torch.rand(num_samples,dim) * X
    ts = torch.randint(low=0,high=num_time,size=(num_samples,1))
    dataset = torch.cat((xs,ts),dim=1)
    data_train = dataset[:20000,:]
    data_val = dataset[20000:,:]
    
    train_kwargs = {'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 1}

    test_kwargs = {'batch_size': batch_size,
            'shuffle': False,
            'num_workers': 1}

    train_loader = torch.utils.data.DataLoader(data_train,**train_kwargs)
    val_loader = torch.utils.data.DataLoader(data_val, **test_kwargs)

    model = FKModule(m=m, dim=dim, p=p, batch_size = batch_size, lr=1e-3, X=X, T=T, N=N, num_time=num_time)
    trainer = pl.Trainer(max_epochs=50,gpus=0)
    trainer.fit(model, train_loader, val_loader)
    
    print(trainer.logged_metrics['val_loss'])
    print(trainer.logged_metrics['train_loss'])