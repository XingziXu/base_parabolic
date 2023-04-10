import torch
from torch import nn
from torch.autograd import grad
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
#from torchvision import datasets, transforms
import numpy as np

#from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
import pytorch_lightning as pl
#from pytorch_lightning.trainer.supporters import CombinedLoader

#from torchvision.utils import make_grid
#import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from sklearn.datasets import make_swiss_roll
#from sklearn.datasets import make_moons
import sys
from random import randint
import seaborn as sns 
import pandas as pd
from torchqrnn import QRNN
import time

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
        h_1 = torch.tanh(self.input_fc(x))
        h_2 = torch.tanh(self.hidden_fc(h_1))
        y_pred = self.output_fc(h_2)
        return y_pred


class FKModule(pl.LightningModule):
    def __init__(self, m=100, dim=10, p=15, batch_size = 100, lr=1e-30, X=1.0, T=1.0, N=1000, num_time=50, n_batch_val=100):
        super().__init__()
        self.num_time = num_time
        self.N = N # number of instances we estimate our PDE solutions with
        self.X = X # interval size
        self.T = T # time interval size
        self.m = m # number of "sensors" for the function u
        self.dim = dim # number of dimension of x
        self.p = p # number of "branches"
        self.lr = 1e-30 # learning rate
        self.sensors = initial((torch.linspace(0., 1., self.m).unsqueeze(-1).repeat(1,self.dim) * self.X).to(device))
        self.batch_size = batch_size
        
        self.branch = MLP(input_dim=self.m, hidden_dim=100, output_dim=self.p) # branch network
        self.trunk = MLP(input_dim=dim+1, hidden_dim=50, output_dim=self.p) # trunk network
        self.branch.load_state_dict(torch.load('/scratch/xx84/girsanov/pde_rnn/branch_5d.pt'))
        self.trunk.load_state_dict(torch.load('/scratch/xx84/girsanov/pde_rnn/trunk_5d.pt'))

        # define the learning rate
        self.lr = lr
                
        # define number of paths used and grid of PDE
        self.N = N
        self.t = torch.linspace(0,1,steps=self.num_time)* self.T
        self.dt = self.t[1]-self.t[0] # define time step

        # define the brwonian motion starting at zero
        self.dB = np.sqrt(self.dt.item()) * np.random.randn(self.t.shape[0], self.N, self.batch_size, self.dim)
        self.dB[0,:,:,:] = 0 
        self.B0 = self.dB.copy()
        self.B0 = torch.Tensor(self.B0.cumsum(0)).to(device)
        self.dB = torch.Tensor(self.dB).to(device)
        
        self.metrics = torch.zeros((1,n_batch_val))
        self.gir_metrics = torch.zeros((1,n_batch_val))
        self.comp_time = torch.zeros((1,n_batch_val))
        self.gir_comp_time = torch.zeros((1,n_batch_val))
        self.rnn_comp_time = torch.zeros((1,n_batch_val))
        self.epochs = torch.linspace(0,0,1)
        
        self.relu = torch.nn.Softplus()

        self.coef_train = torch.rand(10,1,1,3).to(device)
        
        self.relu = torch.nn.Softplus()

    def loss(self, xt, coef):
        xs = xt[:,:-1]
        ts = xt[:,-1]
        coef = coef
        Bx = (xs.unsqueeze(0).unsqueeze(0)+self.B0)
        p0Bx = initial(Bx)
        # calculate values using euler-maruyama
        start = time.time()
        x = torch.zeros(self.num_time, self.N, batch_size, self.dim).to(device)
        x[0,:,:,:] = xs.squeeze()
        for i in range(self.num_time-1):
            x[i+1,:,:,:] = x[i,:,:,:] + drift(x[i,:,:,:], coef).squeeze() * self.dt + self.dB[i,:,:,:]
        p0mux = initial(x)
        u_em = p0mux.mean(1)
        end = time.time()
        time_em = (end - start)
        # calculate values using girsanov
        start = time.time()
        muBx = drift(Bx, coef)
        expmart = torch.exp((torch.cumsum(muBx*self.dB,dim=0) - 0.5 * torch.cumsum((muBx ** 2) * self.dt,dim=0)).sum(-1))
        u_gir = (p0Bx*expmart).mean(1)
        end = time.time()
        time_gir = (end - start)
        # calculate values using deeponet
        start = time.time()
        #branchs = self.branch(self.sensors.unsqueeze(0)).repeat(self.batch_size, 1)
        #u_don = torch.zeros_like(u_em)
        #for i in range(self.num_time-1):
        #    trunks = self.trunk(torch.cat((xs,i*self.dt*torch.ones(xs.shape[0],1).to(device)),dim=1))
        #    u_don[i,:] = (branchs * trunks).sum(1)
        #end = time.time()
        #time_rnn = (end - start)
        branchs = self.branch(self.sensors.unsqueeze(0)).repeat(self.batch_size, 1)
        u_don = torch.zeros_like(u_em)
        for i in range(self.num_time-1):
            trunks = self.trunk(torch.cat((xs,i*self.dt*torch.ones(xs.shape[0],1).to(device)),dim=1))
            u_don[i,:] = (branchs * trunks).sum(1)
        time_rnn = (end - start)
        
        return u_em, u_gir, u_don, time_em, time_gir, time_rnn

    def training_step(self, batch, batch_idx):
        # REQUIRED
        xt = batch.to(device)
        u_em, u_gir, u_rnn, time_em, time_gir, time_rnn = self.loss(xt, coef=torch.rand(1,1,1,3).to(device))
        loss = F.l1_loss(u_rnn, u_gir)
        #tensorboard_logs = {'train_loss': loss_prior}
        self.log('train_loss', loss)
        #print(loss_total)
        return {'loss': loss}
        
        
    def validation_step(self, batch, batch_idx):
        xt = batch.to(device)
        u_em, u_gir, u_rnn, time_em, time_gir, time_rnn = self.loss(xt, coef=torch.rand(1,1,1,3).to(device))
        loss = torch.norm((u_rnn-u_em))/torch.norm(u_em)
        loss_g = torch.norm((u_gir-u_em))/torch.norm(u_em)
        print('Validation: {:.4f}, {:.4f}'.format(loss, loss_g))
        self.log('val_loss', loss)
        if not loss.isnan():
            self.metrics[self.current_epoch, batch_idx] = loss.item()
            self.gir_metrics[self.current_epoch, batch_idx] = loss_g.item()
            self.comp_time[self.current_epoch, batch_idx] = time_em
            self.gir_comp_time[self.current_epoch, batch_idx] = time_gir
            self.rnn_comp_time[self.current_epoch, batch_idx] = time_rnn
            self.log('gir_loss_min', self.gir_metrics[np.where(self.metrics!=0)].min())
            self.log('gir_loss_mean', self.gir_metrics[np.where(self.metrics!=0)].mean())
            self.log('gir_loss_max', self.gir_metrics[np.where(self.metrics!=0)].max())
            self.log('rnn_loss_min', self.metrics[np.where(self.gir_metrics!=0)].min())
            self.log('rnn_loss_mean', self.metrics[np.where(self.gir_metrics!=0)].mean())
            self.log('rnn_loss_max', self.metrics[np.where(self.gir_metrics!=0)].max())
        ep = torch.arange(self.metrics.shape[0])
        plt.plot(ep, self.metrics.mean(-1), label='CNN')
        plt.fill_between(ep, self.metrics.mean(-1) - self.metrics.std(-1), self.metrics.mean(-1) + self.metrics.std(-1), alpha=0.2)
        plt.plot(ep, self.gir_metrics.mean(-1), label='Direct Girsanov')
        plt.fill_between(ep, self.gir_metrics.mean(-1) - self.gir_metrics.std(-1), self.gir_metrics.mean(-1) + self.gir_metrics.std(-1), alpha=0.2)
        plt.ylabel('Relative Error')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig('muBx_2d_cnn_gir.png')
        plt.clf()
        plt.plot(ep, self.metrics.mean(-1), label='CNN')
        plt.fill_between(ep, self.metrics.mean(-1) - self.metrics.std(-1), self.metrics.mean(-1) + self.metrics.std(-1), alpha=0.2)
        plt.ylabel('Relative Error')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig('muBx_2d_cnn.png')
        plt.clf()
        plt.plot(ep, self.comp_time.mean(-1), label='EM')
        plt.fill_between(ep, self.comp_time.mean(-1) - self.comp_time.std(-1), self.comp_time.mean(-1) + self.comp_time.std(-1), alpha=0.2)
        plt.plot(ep, self.gir_comp_time.mean(-1), label='Direct Girsanov')
        plt.fill_between(ep, self.gir_comp_time.mean(-1) - self.gir_comp_time.std(-1), self.gir_comp_time.mean(-1) + self.gir_comp_time.std(-1), alpha=0.2)
        plt.plot(ep, self.rnn_comp_time.mean(-1), label='CNN')
        plt.fill_between(ep, self.rnn_comp_time.mean(-1) - self.rnn_comp_time.std(-1), self.rnn_comp_time.mean(-1) + self.rnn_comp_time.std(-1), alpha=0.2)
        plt.ylabel('Computation Time')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig('comp_time_rnn.png')
        plt.clf()
        #torch.save(self.sequence.state_dict(), 'cnn_5d.pt')
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
    device = torch.device("cuda:0")
    em_min = []
    em_mean = []
    em_max = []
    gir_min = []
    gir_mean = []
    gir_max = []
    don_min = []
    don_mean = []
    don_max = []
    
    for i in range(1,19):
        X = 0.5
        T = i * 0.1
        num_time = 50 * i
        dim = 10
        num_samples = 200
        batch_size = 10
        N = 500
        xs = torch.rand(num_samples,dim) * X
        ts = torch.rand(num_samples,1) * T
        dataset = torch.cat((xs,ts),dim=1)
        data_train = dataset[:2,:]
        data_val = dataset[:,:]
        
        train_kwargs = {'batch_size': batch_size,
                'shuffle': True,
                'num_workers': 1}

        test_kwargs = {'batch_size': batch_size,
                'shuffle': False,
                'num_workers': 1}

        n_batch_val = int(num_samples / batch_size)

        train_loader = torch.utils.data.DataLoader(data_train,**train_kwargs)
        val_loader = torch.utils.data.DataLoader(data_val, **test_kwargs)

        model = FKModule(X=X, T=T, batch_size=batch_size, dim=dim, num_time=num_time, N=N, n_batch_val=n_batch_val)
        trainer = pl.Trainer(max_epochs=1, gpus=1, check_val_every_n_epoch=1)
        trainer.fit(model, train_loader, val_loader)
        
        gir_min.append(trainer.logged_metrics['gir_loss_min'].item())
        gir_mean.append(trainer.logged_metrics['gir_loss_mean'].item())
        gir_max.append(trainer.logged_metrics['gir_loss_max'].item())
        don_min.append(trainer.logged_metrics['rnn_loss_min'].item())
        don_mean.append(trainer.logged_metrics['rnn_loss_mean'].item())
        don_max.append(trainer.logged_metrics['rnn_loss_max'].item())
        #print(trainer.logged_metrics['val_loss'])
        #print(trainer.logged_metrics['train_loss'])
    with open('/scratch/xx84/girsanov/pde_rnn/cnn_loss_mean.npy', 'rb') as f:
        cnn_mean = np.load(f)
    with open('/scratch/xx84/girsanov/pde_rnn/cnn_loss_min.npy', 'rb') as f:
        cnn_min = np.load(f)
    with open('/scratch/xx84/girsanov/pde_rnn/cnn_loss_max.npy', 'rb') as f:
        cnn_max = np.load(f)
    ep = torch.arange(18)
    plt.plot(ep, np.array(em_mean), label='EM')
    plt.fill_between(ep, np.array(em_min), np.array(em_max), alpha=0.2)
    plt.plot(ep, np.array(gir_mean), label='Direct Girsanov')
    plt.fill_between(ep, np.array(gir_min), np.array(gir_max), alpha=0.2)
    plt.plot(ep, np.array(don_mean), label='DeepONet')
    plt.fill_between(ep, np.array(don_min), np.array(don_max), alpha=0.2)
    plt.plot(ep, np.array(cnn_mean), label='CNN')
    plt.fill_between(ep, np.array(cnn_min), np.array(cnn_max), alpha=0.2)
    plt.ylabel('Loss')
    plt.xlabel('Terminal Time')
    plt.legend()
    plt.savefig('loss_don.png')
    plt.clf()
