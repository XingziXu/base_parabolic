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

def drift(x, coef, dim):
    x = x.unsqueeze(-1)
    #x = x/dim
    x0 = (x ** 0)
    x1 = (x ** 1)
    x2 = (x ** 2)
    vals = torch.cat((x0,x1,x2),axis=-1)
    return (coef * vals).sum(-1)

def diffusion(x,t):
    return 1

def initial(x):
    return torch.sin(6*np.pi*x).sum(-1)

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

        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.hidden_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h_1 = torch.tanh(self.input_fc(x))
        h_2 = torch.tanh(self.hidden_fc1(h_1))
        h_3 = torch.tanh(self.hidden_fc2(h_2))
        h_4 = torch.tanh(self.hidden_fc3(h_3))
        y_pred = self.output_fc(h_4)
        return y_pred


class FKModule(pl.LightningModule):
    def __init__(self, m=100, dim=10, p=15, batch_size = 100, lr=1e-3, X=1.0, T=1.0, N=1000, num_time=50, n_batch_val=100):
        super().__init__()
        self.num_time = num_time
        self.N = N # number of instances we estimate our PDE solutions with
        self.X = X # interval size
        self.T = T # time interval size
        self.m = m # number of "sensors" for the function u
        self.dim = dim # number of dimension of x
        self.p = p # number of "branches"
        self.lr = lr # learning rate
        self.sensors = initial((torch.linspace(0., 1., self.m).unsqueeze(-1).repeat(1,self.dim) * self.X).to(device))
        self.batch_size = batch_size
        
        self.branch = MLP(input_dim=self.m, hidden_dim=70+5*dim, output_dim=self.p) # branch network
        self.trunk = MLP(input_dim=dim+1, hidden_dim=50+5*dim, output_dim=self.p) # trunk network

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
        
        self.don_metrics = torch.zeros((50,n_batch_val))
        self.gir_metrics = torch.zeros((50,n_batch_val))
        self.epochs = torch.linspace(0,49,50)
        
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
        x = torch.zeros(self.num_time, self.N, batch_size, self.dim).to(device)
        x[0,:,:,:] = xs
        for i in range(self.num_time-1):
            x[i+1,:,:,:] = x[i,:,:,:] + drift(x[i,:,:,:], coef, self.dim) * self.dt + self.dB[i,:,:,:]
        p0mux = initial(x)
        u_em = p0mux.mean(1)
        # calculate values using girsanov
        muBx = drift(Bx, coef, self.dim)
        expmart = torch.exp((torch.cumsum(muBx*self.dB,dim=0) - 0.5 * torch.cumsum((muBx ** 2) * self.dt,dim=0)).sum(-1))
        u_gir = (p0Bx*expmart).mean(1)
        # calculate values using deeponet
        branchs = self.branch(self.sensors.unsqueeze(0)).repeat(self.batch_size, 1)
        u_don = torch.zeros_like(u_em)
        for i in range(self.num_time-1):
            trunks = self.trunk(torch.cat((xs,i*self.dt*torch.ones(xs.shape[0],1).to(device)),dim=1))
            u_don[i,:] = (branchs * trunks).sum(1)
        return u_em, u_gir, u_don

    def training_step(self, batch, batch_idx):
        # REQUIRED
        xt = batch.to(device)
        idx_ = batch_idx % self.coef_train.shape[0]
        u_em, u_gir, u_rnn = self.loss(xt, coef=self.coef_train[idx_].unsqueeze(0).to(device))
        loss = F.l1_loss(u_rnn, u_gir)
        #tensorboard_logs = {'train_loss': loss_prior}
        self.log('train_loss', loss)
        #print(loss_total)
        return {'loss': loss}
        
        
    def validation_step(self, batch, batch_idx):
        xt = batch.to(device)
        u_em, u_gir, u_don = self.loss(xt, coef=torch.rand(1,1,1,3).to(device))
        loss_don = torch.norm((u_don-u_em))/torch.norm(u_em)
        loss_gir = torch.norm((u_gir-u_em))/torch.norm(u_em)
        print('Validation: {:.4f}, {:.4f}'.format(loss_don, loss_gir))
        self.log('val_loss', loss_don)
        if not loss_don.isnan():
            self.don_metrics[self.current_epoch, batch_idx] = loss_don.item()
            self.gir_metrics[self.current_epoch, batch_idx] = loss_gir.item()
        ep = torch.arange(self.don_metrics.shape[0])
        plt.plot(ep, self.don_metrics.mean(-1), label='DeepONet')
        plt.fill_between(ep, self.don_metrics.mean(-1) - self.don_metrics.std(-1), self.don_metrics.mean(-1) + self.don_metrics.std(-1), alpha=0.2)
        plt.plot(ep, self.gir_metrics.mean(-1), label='Direct Girsanov')
        plt.fill_between(ep, self.gir_metrics.mean(-1) - self.gir_metrics.std(-1), self.gir_metrics.mean(-1) + self.gir_metrics.std(-1), alpha=0.2)
        plt.ylabel('Relative Error')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig('/scratch/xx84/girsanov/fk/ablation/figure/don_train_girloss_full_'+str(self.dim)+'.png')
        plt.clf()
        plt.plot(ep, self.don_metrics.mean(-1), label='DeepONet')
        plt.fill_between(ep, self.don_metrics.mean(-1) - self.don_metrics.std(-1), self.don_metrics.mean(-1) + self.don_metrics.std(-1), alpha=0.2)
        plt.ylabel('Relative Error')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig('/scratch/xx84/girsanov/fk/ablation/figure/don_train_girloss_don_'+str(self.dim)+'.png')
        plt.clf()
        torch.save(self.branch.state_dict(), '/scratch/xx84/girsanov/fk/ablation/trained_model/branch_'+str(self.dim)+'.pt')
        torch.save(self.trunk.state_dict(), '/scratch/xx84/girsanov/fk/ablation/trained_model/trunk_'+str(self.dim)+'.pt')
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
    
    for dim in range(1,2):
    
        m=100
        p=15
        x0 = 0.1
        X = 0.5
        T = 0.1
        num_time = 40
        num_samples = 12000
        batch_size = 40
        N = 4000
        xs = torch.rand(num_samples,dim) * X + x0
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

        model = FKModule(m=m, dim=dim, p=p, batch_size = batch_size, lr=1e-3, X=X, T=T, N=N, num_time=num_time, n_batch_val=n_batch_val)
        trainer = pl.Trainer(max_epochs=10, gpus=1, check_val_every_n_epoch=1)
        trainer.fit(model, train_loader, val_loader)
        
        print(trainer.logged_metrics['val_loss'])
        print(trainer.logged_metrics['train_loss'])
