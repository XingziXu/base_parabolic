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
    def __init__(self, N = 1000, lr = 1e-3, X = 1., T = 0.1, dim = 2, batch_size = 100, num_time = 100, n_batch_val=100):
        super().__init__()
        # define normalizing flow to model the conditional distribution rho(x,t)=p(y|x,t)
        self.num_time = num_time
        self.T = T
        self.t = torch.linspace(0,1,steps=self.num_time)* self.T
        self.dim = dim
        # input size is dimension of brownian motion x 2, since the input to the RNN block is W_s^x and dW_s^x
        input_size = self.dim * 2 + 1
        # hidden_size is dimension of the RNN output
        hidden_size = 50
        # num_layers is the number of RNN blocks
        num_layers = 2
        # num_outputs is the number of ln(rho(x,t))
        num_outputs = self.dim
        self.sequence = RNN(input_size, hidden_size, num_layers, num_outputs)
        #self.sequence.load_state_dict(torch.load('/scratch/xx84/girsanov/pde_rnn/rnn_prior.pt'))

        # define the learning rate
        self.lr = lr
                
        # define number of paths used and grid of PDE
        self.N = N
        self.dt = self.t[1]-self.t[0] # define time step

        # define the brwonian motion starting at zero
        self.dB = np.sqrt(self.dt.item()) * np.random.randn(self.t.shape[0], self.N, batch_size, self.dim)
        self.dB[0,:,:,:] = 0 
        self.B0 = self.dB.copy()
        self.B0 = torch.Tensor(self.B0.cumsum(0)).to(device)
        self.dB = torch.Tensor(self.dB).to(device)
        
        self.metrics = torch.zeros((50,n_batch_val))
        self.gir_metrics = torch.zeros((50,n_batch_val))
        self.epochs = torch.linspace(0,49,50)
        
        self.relu = torch.nn.Softplus()

        self.coef_train = torch.rand(10,1,1,3).to(device)

    def loss(self, xt, coef, return_em=True):
        xs = xt[:,:-1]
        ts = xt[:,-1]
        coef = coef
        Bx = (xs.unsqueeze(0).unsqueeze(0)+self.B0)
        p0Bx = initial(Bx)
        # calculate values using euler-maruyama

        if return_em:
            x = torch.zeros(self.num_time, self.N, batch_size, self.dim).to(device)
            x[0,:,:,:] = xs.squeeze()
            for i in range(self.num_time-1):
                x[i+1,:,:,:] = x[i,:,:,:] + drift(x[i,:,:,:], coef).squeeze() * self.dt + self.dB[i,:,:,:]
            p0mux = initial(x)
            u_em = p0mux.mean(1)
        else:
            u_em = 0 

        # calculate values using girsanov
        muBx = drift(Bx, coef)
        expmart = torch.exp((torch.cumsum(muBx*self.dB,dim=0) - 0.5 * torch.cumsum((muBx ** 2) * self.dt,dim=0)).sum(-1))
        u_gir = (p0Bx*expmart).mean(1)
        # calculate values using RNN
        input = torch.cat((muBx,self.dB,self.dt*torch.ones(self.dB.shape[0],self.dB.shape[1],self.dB.shape[2],1).to(device)),dim=-1)
        input_reshaped = input.reshape(input.shape[1]*input.shape[2], input.shape[0], input.shape[3])
        rnn_expmart = self.relu(self.sequence(input_reshaped).sum(-1)).reshape(p0Bx.shape)
        u_rnn = (p0Bx*rnn_expmart).mean(1)
        return u_em, u_gir, u_rnn

    def training_step(self, batch, batch_idx):
        # REQUIRED
        xt = batch.to(device)
        #u_em, u_gir, u_rnn = self.loss(xt, coef=torch.rand(1,1,1,3).to(device), return_em=False)
        idx_ = batch_idx % self.coef_train.shape[0]
        u_em, u_gir, u_rnn = self.loss(xt, coef=self.coef_train[idx_].unsqueeze(0).to(device), return_em=False)
        loss = F.l1_loss(u_rnn, u_gir)
        #tensorboard_logs = {'train_loss': loss_prior}
        self.log('train_loss', loss)
        #print(loss_total)
        return {'loss': loss}
        
        
    def validation_step(self, batch, batch_idx):
        xt = batch.to(device)
        idx_ = batch_idx % self.coef_train.shape[0]
        u_em, u_gir, u_rnn = self.loss(xt, coef=self.coef_train[idx_].unsqueeze(0).to(device))
        loss = torch.norm((u_rnn-u_em))/torch.norm(u_em)
        loss_g = torch.norm((u_gir-u_em))/torch.norm(u_em)
        print('Validation: {:.4f}, {:.4f}'.format(loss, loss_g))
        self.log('val_loss', loss)
        if not loss.isnan():
            self.metrics[self.current_epoch, batch_idx] = loss.item()
            self.gir_metrics[self.current_epoch, batch_idx] = loss_g.item()
        ep = torch.arange(self.metrics.shape[0])
        plt.plot(ep, self.metrics.mean(-1), label='RNN')
        plt.fill_between(ep, self.metrics.mean(-1) - self.metrics.std(-1), self.metrics.mean(-1) + self.metrics.std(-1), alpha=0.2)
        plt.plot(ep, self.gir_metrics.mean(-1), label='Direct Girsanov')
        plt.fill_between(ep, self.gir_metrics.mean(-1) - self.gir_metrics.std(-1), self.gir_metrics.mean(-1) + self.gir_metrics.std(-1), alpha=0.2)
        plt.ylabel('Relative Error')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig('muBx_2d.png')
        plt.clf()
        plt.plot(ep, self.metrics.mean(-1), label='RNN')
        plt.fill_between(ep, self.metrics.mean(-1) - self.metrics.std(-1), self.metrics.mean(-1) + self.metrics.std(-1), alpha=0.2)
        plt.ylabel('Relative Error')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig('muBx_2d_rnn.png')
        plt.clf()
        #torch.save(self.sequence.state_dict(), 'rnn_10d.pt')
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
    device = torch.device("cuda:0")
    
    X = 0.5
    T = 0.2
    num_time = 100
    dim = 10
    num_samples = 5000
    batch_size = 10
    N = 1000
    xs = torch.rand(num_samples,dim) * X
    ts = torch.rand(num_samples,1) * T
    dataset = torch.cat((xs,ts),dim=1)
    data_train = dataset[:num_samples// 2,:]
    data_val = dataset[num_samples //2 :,:]
    
    train_kwargs = {'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 1}

    test_kwargs = {'batch_size': batch_size,
            'shuffle': False,
            'num_workers': 1}

    n_batch_val = int(num_samples // 2 / batch_size)

    train_loader = torch.utils.data.DataLoader(data_train,**train_kwargs)
    val_loader = torch.utils.data.DataLoader(data_val, **test_kwargs)

    model = FKModule(X=X, T=T, batch_size=batch_size, dim=dim, num_time=num_time, N=N, n_batch_val=n_batch_val)
    trainer = pl.Trainer(max_epochs=50, gpus=1, check_val_every_n_epoch=1)
    trainer.fit(model, train_loader, val_loader)
    
    print(trainer.logged_metrics['val_loss'])
    print(trainer.logged_metrics['train_loss'])
