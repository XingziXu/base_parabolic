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
D_IDX = -1
T_IDX = -2
N_IDX =  0

def drift(x, coef):
    x = x.unsqueeze(-1)
    x0 = x ** 0
    x1 = torch.sin(1 * np.pi * x)
    x2 = torch.sin(2 * np.pi * x)
    x3 = torch.sin(3 * np.pi * x)
    vals = torch.cat((x0,x1,x2,x3),axis=-1)
    return (coef * vals).sum(-1)

def diffusion(x,t):
    return 1
def initial(x):
    return torch.sin(6*np.pi*x)
def initial_val(x):
    return torch.sin(1*np.pi*x)
def r_value():
    return 1

def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.savefig('/scratch/xx84/girsanov/pde_rnn/rnn.png')
    plt.clf()

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
        h_1 = F.tanh(self.input_fc(x))
        h_2 = F.tanh(self.hidden_fc(h_1))
        y_pred = self.output_fc(h_2)
        return y_pred

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_outputs):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        
        # or:
        #self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_outputs)
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        # x: (n, seq, input_size), h0: (num_layers, n, hidden_size)
        
        # Forward propagate RNN
        out, _ = self.gru(x.to(h0), h0)  
        # or:
        #out, _ = self.lstm(x, (h0,c0))  
        
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        #out = out[:, -1, :]
        # out: (n, hidden_size)
         
        out = self.fc(out)
        # out: (n, dimension of rho(x,t))
        return out


class FKModule(pl.LightningModule):
    def __init__(self, N = 500, lr = 1e-3, X = 1., T = 0.1, batch_size = 100):
        super().__init__()
        # define normalizing flow to model the conditional distribution rho(x,t)=p(y|x,t)
        self.num_time = 200
        self.T = T
        self.t = torch.linspace(0,1,steps=self.num_time)* self.T

        # input size is dimension of brownian motion x 2, since the input to the RNN block is W_s^x and dW_s^x
        input_size = 3
        # hidden_size is dimension of the RNN output
        hidden_size = 20
        # num_layers is the number of RNN blocks
        num_layers = 1
        # num_outputs is the number of ln(rho(x,t))
        num_outputs = 1
        self.sequence = RNN(input_size, hidden_size, num_layers, num_outputs)
        self.sequence.load_state_dict(torch.load('/scratch/xx84/girsanov/pde_rnn/rnn_3.pt'))

        # define the learning rate
        self.lr = lr
                
        # define number of paths used and grid of PDE
        self.N = N
        self.dt = self.t[1]-self.t[0] # define time step

        # define the brwonian motion starting at zero
        self.dB = np.sqrt(self.dt.item()) * np.random.randn(self.t.shape[0], self.N, batch_size)
        self.dB[:,:,0] = 0 
        self.B0 = self.dB.copy()
        self.B0 = torch.Tensor(self.B0.cumsum(0)).to(device)
        self.dB = torch.Tensor(self.dB).to(device)
        
        self.metrics = torch.zeros((10,1))
        self.epochs = torch.linspace(0,9,10)
        self.relu = torch.nn.ReLU()

    def loss(self, xt, coef):
        xs = xt[:,0]
        ts = xt[:,1]
        coef = coef
        Bx = (xs.unsqueeze(0).unsqueeze(0)+self.B0)
        p0Bx = initial(Bx)
        # calculate values using euler-maruyama
        x = torch.zeros(self.num_time, self.N, batch_size).to(device)
        x[0,:,:] = xs.squeeze()
        for i in range(self.num_time-1):
            x[i+1,:,:] = x[i,:,:] + drift(x[i,:,:], coef).squeeze() * self.dt + self.dB[i,:,:]
        p0mux = initial(x)
        u_em = p0mux.mean(1)
        # calculate values using girsanov
        muBx = drift(Bx, coef)
        expmart = torch.exp(torch.cumsum(muBx*self.dB,dim=0) - 0.5 * torch.cumsum((muBx ** 2) * self.dt,dim=0))
        u_gir = (p0Bx*expmart).mean(1)
        # calculate values using RNN
        input = torch.cat((muBx.unsqueeze(-1),self.dB.unsqueeze(-1),self.dt*torch.ones_like(Bx).unsqueeze(-1)),dim=-1)
        input_reshaped = input.reshape(input.shape[1]*input.shape[2], input.shape[0], input.shape[3])
        rnn_expmart = self.relu(self.sequence(input_reshaped).reshape(p0Bx.shape))
        u_rnn = (p0Bx*rnn_expmart).mean(1)
        return u_em, u_gir, u_rnn

    def training_step(self, batch, batch_idx):
        # REQUIRED
        xt = batch.to(device)
        u_em, u_gir, u_rnn = self.loss(xt, coef=torch.rand(1,1,1,4).to(device))
        loss = torch.norm((u_rnn-u_gir))/torch.norm(u_gir)
        #tensorboard_logs = {'train_loss': loss_prior}
        self.log('train_loss', loss)
        #print(loss_total)
        return {'loss': loss}
        
        
    def validation_step(self, batch, batch_idx):
        #xtu = batch.to(device)
        #loss_total = self.loss(xtu)
        #self.log('val_loss', loss_total)
        #print(loss_total)
        #if torch.rand(1)[0]>0.8:
        xt = batch.to(device)
        u_em, u_gir, u_rnn = self.loss(xt, coef=torch.rand(1,1,1,4).to(device))
        loss = torch.norm((u_rnn-u_em))/torch.norm(u_em)
        #tensorboard_logs = {'train_loss': loss_prior}
        self.log('val_loss', loss)
        #print(loss_total)
        self.metrics[self.current_epoch,:] = loss.item()
        plt.plot(self.epochs, self.metrics[:,0], label='Val_loss')
        plt.ylabel('Magnitude')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig('/scratch/xx84/girsanov/pde_rnn/muBx.png')
        plt.clf()
        
        plt.plot(u_em[30,:].cpu(), label='em')
        plt.plot(u_gir[30,:].cpu(), label='girsanov')
        plt.plot(u_rnn[30,:].cpu(), label='rnn')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.savefig('/scratch/xx84/girsanov/pde_rnn/xts.png')
        plt.clf()
        return #{'loss': loss_total}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.sequence.gru.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}



if __name__ == '__main__':
    pl.seed_everything(1235)
    print(sys.executable)
    #dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    #mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
    #mnist_train, mnist_val = random_split(dataset, [55000,5000])
    device = torch.device("cuda:0")
    
    X = 0.5
    T = 0.02
    num_samples = 50
    batch_size = 50
    xs = torch.linspace(0, 1, 50).unsqueeze(-1) * X
    ts = torch.rand(num_samples,1) * T
    dataset = torch.cat((xs,ts),dim=1)
    data_train = dataset[:,:]
    data_val = dataset[:,:]
    
    train_kwargs = {'batch_size': batch_size,
            'shuffle': False,
            'num_workers': 4}

    test_kwargs = {'batch_size': batch_size,
            'shuffle': False,
            'num_workers': 4}

    train_loader = torch.utils.data.DataLoader(data_train,**train_kwargs)
    val_loader = torch.utils.data.DataLoader(data_val, **test_kwargs)

    model = FKModule(X=X, T=T, batch_size=batch_size)
    trainer = pl.Trainer(max_epochs=10,gpus=1)
    trainer.fit(model, train_loader, val_loader)
    
    print(trainer.logged_metrics['val_loss'])
    print(trainer.logged_metrics['train_loss'])