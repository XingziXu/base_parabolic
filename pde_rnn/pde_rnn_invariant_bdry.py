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

def drift(coef, x, t):
    x = np.expand_dims(x,axis=2)
    t = t * np.ones_like(x)
    x0 = x ** 0
    x1 = x ** 1
    x2 = x ** 2
    x3 = x ** 3
    t1 = t ** 1
    t2 = t ** 2
    vals = np.concatenate((x0,x1,x2,x3,t1,t2),axis=2)
    return (np.expand_dims(coef.numpy(), axis=0) * vals).sum(-1)
def diffusion(x,t):
    return 1
def initial0(x,t):
    return np.sin(6*np.pi*x)
def initial1(x,t):
    return np.sin(1*np.pi*x)
def initial2(x,t):
    return np.sin(3*np.pi*x) * t
def initial3(x,t):
    return np.cos(6*np.pi*x)
def initial4(x,t):
    return np.cos(2*np.pi*x) * np.sqrt(t)
def initial_val(x,t):
    return np.sin(1*np.pi*x)
def r_value():
    return 1


class PDETask:
    def __init__(self, num_data=int(3000), coef = torch.rand(1, 6) * 0.3):
        self.num_data = num_data
        self.num_path = 1000 # simulate 500 runs of brownian motion
        self.num_time = 50 # simulate brownian motion with 500 time steps
        self.num_pos = 200 # simulate the space in 100 time steps
        self.t_init = 0
        self.t_end = 0.01 # define the ending time
        self.x_init = 0
        self.x_end = 1. # define the ending position
        self.t = np.linspace(self.t_init,self.t_end,num=self.num_time) # define the time
        self.x_init = np.linspace(self.x_init,self.x_end,num=self.num_pos) # define the position
        self.dt = self.t[1]-self.t[0] # define time step
        self.dB = np.sqrt(self.dt) * np.random.randn(self.num_path,self.num_pos,self.num_time)
        self.dB[:,:,0] = 0
        self.coef = coef


    def sample(self):
        xs = np.zeros((self.num_path,self.num_pos,self.num_time))
        xs[:,:,0] = self.x_init.copy()
        ts = np.arange(self.t_init, self.t_end + self.dt, self.dt)
        ts_expanded = np.repeat(np.repeat(ts[np.newaxis,...], self.num_pos,axis=0)[np.newaxis,...], self.num_path,axis=0)
        #output = np.zeros((num_path,num_pos,num_time))
        for i in range(1, ts.size):
            t = self.t_init + (i - 1) * self.dt
            x = xs[:,:,i-1]
            xs[:,:,i] = x + drift(self.coef, x, t) * self.dt + diffusion(x, t) * self.dB[:,:,i]
        exponential=np.exp(-r_value()*np.repeat(ts[np.newaxis,...], self.num_pos,axis=0))
        ini_expectation = np.mean(initial0(xs,ts_expanded),axis=0)
        data0 = exponential * ini_expectation
        
        # xs = np.zeros((self.num_path,self.num_pos,self.num_time))
        # xs[:,:,0] = self.x_init.copy()
        # ts = np.arange(self.t_init, self.t_end + self.dt, self.dt)
        # ts_expanded = np.repeat(np.repeat(ts[np.newaxis,...], self.num_pos,axis=0)[np.newaxis,...], self.num_path,axis=0)
        # #output = np.zeros((num_path,num_pos,num_time))
        # for i in range(1, ts.size):
        #     t = self.t_init + (i - 1) * self.dt
        #     x = xs[:,:,i-1]
        #     xs[:,:,i] = x + drift(self.coef, x, t) * self.dt + diffusion(x, t) * self.dB[:,:,i]
        # exponential=np.exp(-r_value()*np.repeat(ts[np.newaxis,...], self.num_pos,axis=0))
        # ini_expectation = np.mean(initial1(xs,ts_expanded),axis=0)
        # data1 = exponential * ini_expectation
        
        # xs = np.zeros((self.num_path,self.num_pos,self.num_time))
        # xs[:,:,0] = self.x_init.copy()
        # ts = np.arange(self.t_init, self.t_end + self.dt, self.dt)
        # ts_expanded = np.repeat(np.repeat(ts[np.newaxis,...], self.num_pos,axis=0)[np.newaxis,...], self.num_path,axis=0)
        # #output = np.zeros((num_path,num_pos,num_time))
        # for i in range(1, ts.size):
        #     t = self.t_init + (i - 1) * self.dt
        #     x = xs[:,:,i-1]
        #     xs[:,:,i] = x + drift(self.coef, x, t) * self.dt + diffusion(x, t) * self.dB[:,:,i]
        # exponential=np.exp(-r_value()*np.repeat(ts[np.newaxis,...], self.num_pos,axis=0))
        # ini_expectation = np.mean(initial2(xs,ts_expanded),axis=0)
        # data2 = exponential * ini_expectation
        
        # xs = np.zeros((self.num_path,self.num_pos,self.num_time))
        # xs[:,:,0] = self.x_init.copy()
        # ts = np.arange(self.t_init, self.t_end + self.dt, self.dt)
        # ts_expanded = np.repeat(np.repeat(ts[np.newaxis,...], self.num_pos,axis=0)[np.newaxis,...], self.num_path,axis=0)
        # #output = np.zeros((num_path,num_pos,num_time))
        # for i in range(1, ts.size):
        #     t = self.t_init + (i - 1) * self.dt
        #     x = xs[:,:,i-1]
        #     xs[:,:,i] = x + drift(self.coef, x, t) * self.dt + diffusion(x, t) * self.dB[:,:,i]
        # exponential=np.exp(-r_value()*np.repeat(ts[np.newaxis,...], self.num_pos,axis=0))
        # ini_expectation = np.mean(initial3(xs,ts_expanded),axis=0)
        # data3 = exponential * ini_expectation
        
        # xs = np.zeros((self.num_path,self.num_pos,self.num_time))
        # xs[:,:,0] = self.x_init.copy()
        # ts = np.arange(self.t_init, self.t_end + self.dt, self.dt)
        # tts_expanded = np.repeat(np.repeat(ts[np.newaxis,...], self.num_pos,axis=0)[np.newaxis,...], self.num_path,axis=0)
        # #output = np.zeros((num_path,num_pos,num_time))
        # for i in range(1, ts.size):
        #     t = self.t_init + (i - 1) * self.dt
        #     x = xs[:,:,i-1]
        #     xs[:,:,i] = x + drift(self.coef, x, t) * self.dt + diffusion(x, t) * self.dB[:,:,i]
        # exponential=np.exp(-r_value()*np.repeat(ts[np.newaxis,...], self.num_pos,axis=0))
        # ini_expectation = np.mean(initial4(xs,ts_expanded),axis=0)
        # data4 = exponential * ini_expectation
        
        
        x_data_idx0 = np.asarray([random.randrange(1, self.num_pos, 1) for i in range(self.num_data)])
        t_data_idx0 = np.asarray([random.randrange(1, self.num_time, 1) for i in range(self.num_data)])
        x_data0 = self.x_init[x_data_idx0]
        t_data0 = ts[t_data_idx0]
        u_data0 = data0[x_data_idx0,t_data_idx0]
        
        # x_data_idx1 = np.asarray([random.randrange(1, self.num_pos, 1) for i in range(self.num_data)])
        # t_data_idx1 = np.asarray([random.randrange(1, self.num_time, 1) for i in range(self.num_data)])
        # x_data1 = self.x_init[x_data_idx1]
        # t_data1 = ts[t_data_idx1]
        # u_data1 = data0[x_data_idx1,t_data_idx1]
        
        # x_data_idx2 = np.asarray([random.randrange(1, self.num_pos, 1) for i in range(self.num_data)])
        # t_data_idx2 = np.asarray([random.randrange(1, self.num_time, 1) for i in range(self.num_data)])
        # x_data2 = self.x_init[x_data_idx2]
        # t_data2 = ts[t_data_idx2]
        # u_data2 = data0[x_data_idx2,t_data_idx2]
        
        # x_data_idx3 = np.asarray([random.randrange(1, self.num_pos, 1) for i in range(self.num_data)])
        # t_data_idx3 = np.asarray([random.randrange(1, self.num_time, 1) for i in range(self.num_data)])
        # x_data3 = self.x_init[x_data_idx3]
        # t_data3 = ts[t_data_idx3]
        # u_data3 = data3[x_data_idx3,t_data_idx3]
        
        # x_data_idx4 = np.asarray([random.randrange(1, self.num_pos, 1) for i in range(self.num_data)])
        # t_data_idx4 = np.asarray([random.randrange(1, self.num_time, 1) for i in range(self.num_data)])
        # x_data4 = self.x_init[x_data_idx4]
        # t_data4 = ts[t_data_idx4]
        # u_data4 = data0[x_data_idx4,t_data_idx4]
        
        # x_data = np.concatenate((x_data0,x_data1,x_data2,x_data3,x_data4))
        # x_data_idx = np.concatenate((x_data_idx0,x_data_idx1,x_data_idx2,x_data_idx3,x_data_idx4))
        # t_data = np.concatenate((t_data0,t_data1,t_data2,t_data3,t_data4))
        # t_data_idx = np.concatenate((t_data_idx0,t_data_idx1,t_data_idx2,t_data_idx3,t_data_idx4))
        # u_data = np.concatenate((u_data0,u_data1,u_data2,u_data3,u_data4))
        total = torch.tensor(np.concatenate((x_data0.reshape(len(x_data0),1),x_data_idx0.reshape(len(x_data_idx0),1),t_data0.reshape(len(t_data0),1),t_data_idx0.reshape(len(t_data_idx0),1),u_data0.reshape(len(u_data0),1)),axis=1))
        return total

class PDETask_val:
    def __init__(self, num_data=int(500), coef = torch.rand(1, 6) * 0.3):
        self.num_data = num_data
        self.num_path = 1000 # simulate 500 runs of brownian motion
        self.num_time = 50 # simulate brownian motion with 500 time steps
        self.num_pos = 200 # simulate the space in 100 time steps
        self.t_init = 0
        self.t_end = 0.01 # define the ending time
        self.x_init = 0
        self.x_end = 1. # define the ending position
        self.t = np.linspace(self.t_init,self.t_end,num=self.num_time) # define the time
        self.x_init = np.linspace(self.x_init,self.x_end,num=self.num_pos) # define the position
        self.dt = self.t[1]-self.t[0] # define time step
        self.dB = np.sqrt(self.dt) * np.random.randn(self.num_path,self.num_pos,self.num_time)
        self.dB[:,:,0] = 0
        self.coef = coef


    def sample(self):
        xs = np.zeros((self.num_path,self.num_pos,self.num_time))
        xs[:,:,0] = self.x_init.copy()
        ts = np.arange(self.t_init, self.t_end + self.dt, self.dt)
        ts_expanded = np.repeat(np.repeat(ts[np.newaxis,...], 200,axis=0)[np.newaxis,...],500,axis=0)
        #output = np.zeros((num_path,num_pos,num_time))
        for i in range(1, ts.size):
            t = self.t_init + (i - 1) * self.dt
            x = xs[:,:,i-1]
            xs[:,:,i] = x + drift(self.coef, x, t) * self.dt + diffusion(x, t) * self.dB[:,:,i]
        exponential=np.exp(-r_value()*np.repeat(ts[np.newaxis,...], 200,axis=0))
        ini_expectation = np.mean(initial_val(xs,ts_expanded),axis=0)
        data = exponential * ini_expectation
        x_data_idx = np.asarray([random.randrange(1, self.num_pos, 1) for i in range(self.num_data)])
        t_data_idx = np.asarray([random.randrange(1, self.num_time, 1) for i in range(self.num_data)])
        x_data = self.x_init[x_data_idx]
        t_data = ts[t_data_idx]
        u_data = data[x_data_idx,t_data_idx]
        total = torch.tensor(np.concatenate((x_data.reshape(len(x_data),1),x_data_idx.reshape(len(x_data_idx),1),t_data.reshape(len(t_data),1),t_data_idx.reshape(len(t_data_idx),1),u_data.reshape(len(u_data),1)),axis=1))
        return total

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
        out = out[:, -1, :]
        # out: (n, hidden_size)
         
        out = self.fc(out)
        # out: (n, dimension of rho(x,t))
        return out


class FKModule(pl.LightningModule):
    def __init__(self, dt = 0.0001, N = 1000, lr = 5e-3):
        super().__init__()
        # define normalizing flow to model the conditional distribution rho(x,t)=p(y|x,t)
        self.num_time = 50
        self.T = 0.01
        #self.T = torch.nn.Parameter(torch.FloatTensor([T0]), requires_grad=False).to(device)
        #self.forward_sde = VariancePreservingSDE(beta_min=0.1, beta_max=20.0, T=self.T).to(device)
        self.t = torch.linspace(0,1,steps=self.num_time)* self.T
        #t = (torch.linspace(0, 1, int(self.T/dt)).to(device)) * self.T
        #self.register_buffer('t', t)
        #self.a = MLP1(input_dim=1, index_dim=1, hidden_dim=156).to(device)

        # input size is dimension of brownian motion x 2, since the input to the RNN block is W_s^x and dW_s^x
        input_size = 3
        # hidden_size is dimension of the RNN output
        hidden_size = 40
        # num_layers is the number of RNN blocks
        num_layers = 2
        # num_outputs is the number of ln(rho(x,t))
        num_outputs = 1
        self.sequence = RNN(input_size, hidden_size, num_layers, num_outputs)
        #self.sequence.load_state_dict(torch.load('/scratch/xx84/girsanov/pde_rnn/rnn_prior.pt'))

        # define the learning rate
        self.lr = lr

        # constant
        self.Log2PI = float(np.log(2 * np.pi))
                
        # define number of paths used and grid of PDE
        self.N = N
        self.num_time = 50 # simulate brownian motion with 500 time steps
        self.num_pos = 200 # simulate the space in 100 time steps
        self.x_init = 0.
        self.x_end = 1. # define the ending position
        self.x = np.linspace(self.x_init,self.x_end,num=self.num_pos) # define the position

        self.dt = self.t[1]-self.t[0] # define time step

        # define the brwonian motion starting at different positions
        self.dB = np.sqrt(self.dt.item()) * np.random.randn(self.N, self.x.shape[0], self.t.shape[0])
        self.dB[:,:,0] = 0 
        self.B = self.dB.copy()
        self.B[:,:,0] = self.x.copy()
        self.B = self.B.cumsum(-1)
        
        self.gwt = initial0(self.B,self.t.numpy())
        self.gwt_val = initial_val(self.B,self.t.numpy())

    def drift(self, x, t):
        coef = torch.Tensor([[0.0087, 0.1206, 0.0780, 0.1100, 0.0175, 0.2102]])
        #x = np.expand_dims(x,axis=2)
        t = t * torch.ones_like(x)
        x0 = x ** 0
        x1 = x ** 1
        x2 = x ** 2
        x3 = x ** 3
        t1 = t ** 1
        t2 = t ** 2
        vals = torch.cat((x0,x1,x2,x3,t1,t2),dim=2)
        return (coef.unsqueeze(0) * vals).sum(-1)

    def loss(self, xtu, gwt):
        # get the values for x and t
        xt = xtu[:,[0,2]] 
        # get the values for u
        u = xtu[:,-1]
        # we generated x, t, dB, and B all according to the same indices, we get the indices here to recover them
        x_idx = xtu[:,1].type(torch.IntTensor)
        t_idx = xtu[:,3].type(torch.IntTensor)
        # for the network, we need four inputs, ds, x, dB, B_current, here we get ds
        dsi = self.dt
        # for every x and t
        u_hat = torch.zeros(len(xtu),1).to(device)
        for i in range(len(xtu)):
            #t_idx_max = t_idx[i]
            # get the values calculated using only brownian motion
            gwt_current = torch.tensor(gwt[:,x_idx[i],t_idx[i]])
            # for time steps from 0 to t
            #for j in range(t_idx_max):
            # get the dB at the current time for brownian motion started at current x
            dB_current = torch.tensor(self.dB[:,x_idx[i],:t_idx[i]]).unsqueeze(2)
            # get the B at the current time for brownian motion started at current x
            #B_current = self.drift(x=torch.tensor(self.B[:,x_idx[i],:t_idx[i]]).unsqueeze(2),t=t_idx[i]*self.dt).unsqueeze(2)
            B_current = torch.tensor(self.B[:,x_idx[i],:t_idx[i]]).unsqueeze(2)
            # get the time and the x
            #    xt_current = xt[i,1].repeat(len(dB_current),1)
            # get the ds
            ds_current = torch.ones(len(dB_current),B_current.shape[1],1)*dsi
            # form the input
            input_current = torch.cat((dB_current, B_current, ds_current),dim=2)
            # estimate rho(x,t) using a recurrent nural network
            rho_est =  torch.exp(self.sequence(input_current.type(torch.FloatTensor)))
            # calculate the value by taking the average
            u_hat[i] = ((gwt_current.unsqueeze(1)).to(rho_est)*rho_est).mean(0)
        # calculate loss
        #loss = torch.norm((u_hat-u.unsqueeze(1))/(t_idx.unsqueeze(1)+1.))
        loss = torch.norm((u_hat-u.unsqueeze(1)))/torch.norm(u.unsqueeze(1))
        #print(torch.norm((u_hat-u.unsqueeze(1)))/torch.norm(u.unsqueeze(1)))
        return loss

    def sample(self, xt_val):
        t_idx = xt_val[:,-2]
        x_idx = xt_val[:,-1]
        xt = xt_val[:,0:2].to(device)
        dsi = self.dt
        u_hat = torch.zeros(len(xt_val),1).to(device)
        for i in range(len(xt_val)):
            #t_idx_max = t_idx[i]
            # get the values calculated using only brownian motion
            gwt_current = torch.tensor(self.gwt[:,int(x_idx[i]),int(t_idx[i])])
            # for time steps from 0 to t
            #for j in range(t_idx_max):
            # get the dB at the current time for brownian motion started at current x
            dB_current = torch.tensor(self.dB[:,int(x_idx[i]),:int(t_idx[i])]).unsqueeze(2)
            # get the B at the current time for brownian motion started at current x
            B_current = torch.tensor(self.B[:,int(x_idx[i]),:int(t_idx[i])]).unsqueeze(2)
            # get the time and the x
            #    xt_current = xt[i,1].repeat(len(dB_current),1)
            # get the ds
            ds_current = torch.ones(len(dB_current),B_current.shape[1],1)*dsi
            # form the input
            input_current = torch.cat((dB_current, B_current, ds_current),dim=2)
            # estimate rho(x,t) using a recurrent nural network
            rho_est =  torch.exp(self.sequence(input_current.type(torch.FloatTensor)))
            # calculate the value by taking the average
            u_hat[i] = ((gwt_current.unsqueeze(1)).to(rho_est)*rho_est).mean(0)
        df_tensor = torch.cat((xt,u_hat),axis=1)
        df = pd.DataFrame(df_tensor.cpu().numpy())
        #df = df.rename(columns={'0': 'X', '1': 'Y', '2': 'val'})  # new method
        table = df.pivot(0, 1, 2)
        ax = sns.heatmap(table)
        ax.invert_yaxis()
        #print(table)
        plt.show()
        plt.savefig('temp.png')
        return 


    def training_step(self, batch, batch_idx):
        # REQUIRED
        xtu = batch.to(device)
        loss_total = self.loss(xtu, self.gwt)
        #tensorboard_logs = {'train_loss': loss_prior}
        self.log('train_loss', loss_total)
        #print(loss_total)
        return {'loss': loss_total}
        
        
    def validation_step(self, batch, batch_idx):
        #xtu = batch.to(device)
        #loss_total = self.loss(xtu)
        #self.log('val_loss', loss_total)
        #print(loss_total)
        #if torch.rand(1)[0]>0.8:
        xtu = batch.to(device)
        loss_total = self.loss(xtu, self.gwt_val)
        #tensorboard_logs = {'train_loss': loss_prior}
        self.log('val_loss', loss_total)
        print(loss_total)
        """
        if self.trainer.current_epoch% 10 == 0:
            x_vals = torch.unsqueeze(torch.tensor(self.x[1:]).repeat_interleave(len(self.t[1:])), 1)
            t_vals = torch.unsqueeze(self.t[1:].repeat(1,len(self.x[1:]))[0], 1)
            t_idx = torch.unsqueeze(torch.linspace(1,len(self.t)-1,steps=len(self.t)-1).repeat(1,len(self.x)-1)[0], 1)
            x_idx = torch.unsqueeze(torch.linspace(1,len(self.x)-1,steps=len(self.x)-1).repeat_interleave(len(self.t)-1), 1)
            xt_val = torch.cat((x_vals, t_vals, t_idx, x_idx), dim=1)
            self.sample(xt_val)
            """
        return #{'loss': loss_total}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.sequence.gru.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}


    #def train_dataloader(self):
    #    task = PDETask()
    #    dataset = task.sample()
    #    train_loader = torch.utils.data.DataLoader(dataset, batch_size = 400, shuffle=True, num_workers = 1)
    #    return train_loader

    #def val_dataloader(self):
    #    task = PDETask()
    #    dataset = task.sample()
    #    val_loader = torch.utils.data.DataLoader(dataset, batch_size = 400, shuffle=True, num_workers = 1)
    #    return val_loader



if __name__ == '__main__':
    pl.seed_everything(1234)
    print(sys.executable)
    #dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    #mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
    #mnist_train, mnist_val = random_split(dataset, [55000,5000])
    device = torch.device("cuda:0")
    

    train_kwargs = {'batch_size': 100,
            'shuffle': True,
            'num_workers': 4}

    test_kwargs = {'batch_size': 100,
            'shuffle': False,
            'num_workers': 4}
    

    #coef = torch.rand(1,6) * 0.3
    coef = torch.Tensor([[0.0087, 0.1206, 0.0780, 0.1100, 0.0175, 0.2102]])
    task = PDETask(coef=coef)
    task_val = PDETask_val(coef=coef)
    dataset1 = task.sample()
    dataset2 = task_val.sample()
    
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    val_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = FKModule()
    trainer = pl.Trainer(max_epochs=10,gpus=1)
    trainer.fit(model, train_loader, val_loader)
    
    print(trainer.logged_metrics['val_loss'])
    print(trainer.logged_metrics['train_loss'])