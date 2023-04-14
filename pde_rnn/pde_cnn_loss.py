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
        h_1 = F.tanh(self.input_fc(x))
        h_2 = F.tanh(self.hidden_fc(h_1))
        y_pred = self.output_fc(h_2)
        return y_pred

class CNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_outputs):
		super(CNN, self).__init__()
		#self.num_layers = num_layers
		#self.hidden_size = hidden_size
		self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=1, padding=0)
		self.act1 = nn.Softplus()
		self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, padding=0)
		self.act2 = nn.Softplus()
		self.conv3 = nn.Conv1d(in_channels=hidden_size, out_channels=num_outputs, kernel_size=1, padding=0)
		self.act3 = nn.Softplus()
	
	def forward(self, x):
		out = self.conv1(x)
		out = self.act1(out)
		out = self.conv2(out)
		out = self.act2(out)
		out = self.conv3(out)
		#out = self.act3(out)
		return out


class FKModule(pl.LightningModule):
    def __init__(self, N = 2000, lr = 1e-30, X = 1., T = 0.1, dim = 2, batch_size = 100, num_time = 100, n_batch_val=100):
        super().__init__()
        # define normalizing flow to model the conditional distribution rho(x,t)=p(y|x,t)
        self.num_time = num_time
        self.T = T
        self.t = torch.linspace(0,1,steps=self.num_time)* self.T
        self.dim = dim
        self.batch_size = batch_size
        # input size is dimension of brownian motion x 2, since the input to the RNN block is W_s^x and dW_s^x
        input_size = self.dim * 2 + 1
        # hidden_size is dimension of the RNN output
        hidden_size = 80
        # num_layers is the number of RNN blocks
        num_layers = 3
        # num_outputs is the number of ln(rho(x,t))
        num_outputs = self.dim
        self.sequence = CNN(input_size, hidden_size, num_layers, num_outputs)
        self.sequence.load_state_dict(torch.load('/scratch/xx84/girsanov/pde_rnn/cnn_5d.pt'))

        # define the learning rate
        self.lr = 1e-30
                
        # define number of paths used and grid of PDE
        self.N = N
        self.dt = self.t[1]-self.t[0] # define time step

        # define the brwonian motion starting at zero
        self.dB = np.sqrt(self.dt.item()) * np.random.randn(self.t.shape[0], self.N, self.batch_size, self.dim)
        self.dB[0,:,:,:] = 0 
        self.B0 = self.dB.copy()
        self.B0 = torch.Tensor(self.B0.cumsum(0)).to(device)
        self.dB = torch.Tensor(self.dB).to(device)
        
        self.cnn_metrics = torch.zeros((1,n_batch_val))
        self.gir_metrics = torch.zeros((1,n_batch_val))
        self.don_metrics = torch.zeros((1,n_batch_val))
        self.em_comp_time = torch.zeros((1,n_batch_val))
        self.gir_comp_time = torch.zeros((1,n_batch_val))
        self.cnn_comp_time = torch.zeros((1,n_batch_val))
        self.don_comp_time = torch.zeros((1,n_batch_val))
        self.epochs = torch.linspace(0,0,1)
        
        self.relu = torch.nn.Softplus()

        self.coef_train = torch.rand(10,1,1,3).to(device)
        
        self.relu = torch.nn.Softplus()
        
        self.m = 100 # number of "sensors" for the function u
        self.p = 15 # number of "branches"
        self.lr = 1e-30 # learning rate
        self.X = X # interval size
        self.sensors = initial((torch.linspace(0., 1., self.m).unsqueeze(-1).repeat(1,self.dim) * self.X).to(device))
        self.branch = MLP(input_dim=self.m, hidden_dim=100, output_dim=self.p) # branch network
        self.trunk = MLP(input_dim=dim+1, hidden_dim=50, output_dim=self.p) # trunk network
        self.branch.load_state_dict(torch.load('/scratch/xx84/girsanov/pde_rnn/branch_5d.pt'))
        self.trunk.load_state_dict(torch.load('/scratch/xx84/girsanov/pde_rnn/trunk_5d.pt'))

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
        # calculate values using RNN
        
        input = torch.zeros(self.num_time, self.N, self.batch_size, self.dim * 2 + 1).to(device)
        input[:muBx.shape[0],:,:,:] = torch.cat((muBx,self.dB,self.dt*torch.ones(self.dB.shape[0],self.dB.shape[1],self.dB.shape[2],1).to(device)),dim=-1)
        input_reshaped = input.reshape(input.shape[1]*input.shape[2], input.shape[3], input.shape[0])
        start = time.time()
        rnn_expmart = self.relu(self.sequence(input_reshaped).sum(-2)).reshape(p0Bx.shape)
        u_cnn = (p0Bx*rnn_expmart).mean(1)
        end = time.time()
        time_cnn = (end - start)
        # calculate values using deeponet
        
        axis = torch.Tensor([]).to(device)
        for i in range(self.num_time):
            current_axis = torch.cat((xs,i*self.dt*torch.ones(xs.shape[0],1).to(device)),dim=1)
            axis = torch.cat((axis,current_axis),dim=0)
        start = time.time()
        branchs = self.branch(self.sensors.unsqueeze(0)).repeat(self.batch_size*(self.num_time), 1)
        u_don = torch.zeros_like(u_em)
        trunks = self.trunk(axis)
        u_don = (branchs * trunks).sum(1).reshape(u_gir.shape)
        end = time.time()
        time_don = (end - start)
        return u_em, u_gir, u_cnn, u_don, time_em, time_gir, time_cnn, time_don

    def training_step(self, batch, batch_idx):
        # REQUIRED
        xt = batch.to(device)
        u_em, u_gir, u_cnn, u_don, time_em, time_gir, time_cnn, time_don = self.loss(xt, coef=torch.rand(1,1,1,3).to(device))
        loss = F.l1_loss(u_cnn, u_gir)
        #tensorboard_logs = {'train_loss': loss_prior}
        self.log('train_loss', loss)
        #print(loss_total)
        return {'loss': loss}
        
        
    def validation_step(self, batch, batch_idx):
        xt = batch.to(device)
        u_em, u_gir, u_cnn, u_don, time_em, time_gir, time_cnn, time_don = self.loss(xt, coef=torch.rand(1,1,1,3).to(device))
        loss_cnn = torch.norm((u_cnn-u_em))/torch.norm(u_em)
        loss_gir = torch.norm((u_gir-u_em))/torch.norm(u_em)
        loss_don = torch.norm((u_don-u_em))/torch.norm(u_em)
        print('Validation: {:.4f}, {:.4f}'.format(loss_cnn, loss_gir))
        self.log('val_loss', loss_cnn)
        if not loss_cnn.isnan():
            self.cnn_metrics[self.current_epoch, batch_idx] = loss_cnn.item()
            self.gir_metrics[self.current_epoch, batch_idx] = loss_gir.item()
            self.don_metrics[self.current_epoch, batch_idx] = loss_don.item()
            self.em_comp_time[self.current_epoch, batch_idx] = time_em
            self.gir_comp_time[self.current_epoch, batch_idx] = time_gir
            self.cnn_comp_time[self.current_epoch, batch_idx] = time_cnn
            self.don_comp_time[self.current_epoch, batch_idx] = time_don
            self.log('gir_loss_min', self.gir_metrics[np.where(self.gir_metrics!=0)].min())
            self.log('gir_loss_mean', self.gir_metrics[np.where(self.gir_metrics!=0)].mean())
            self.log('gir_loss_max', self.gir_metrics[np.where(self.gir_metrics!=0)].max())
            self.log('cnn_loss_min', self.cnn_metrics[np.where(self.cnn_metrics!=0)].min())
            self.log('cnn_loss_mean', self.cnn_metrics[np.where(self.cnn_metrics!=0)].mean())
            self.log('cnn_loss_max', self.cnn_metrics[np.where(self.cnn_metrics!=0)].max())
            self.log('don_loss_min', self.don_metrics[np.where(self.don_metrics!=0)].min())
            self.log('don_loss_mean', self.don_metrics[np.where(self.don_metrics!=0)].mean())
            self.log('don_loss_max', self.don_metrics[np.where(self.don_metrics!=0)].max())
            
            self.log('em_time_min', self.em_comp_time[np.where(self.em_comp_time!=0)].min())
            self.log('em_time_mean', self.em_comp_time[np.where(self.em_comp_time!=0)].mean())
            self.log('em_time_max', self.em_comp_time[np.where(self.em_comp_time!=0)].max())
            self.log('gir_time_min', self.gir_comp_time[np.where(self.gir_comp_time!=0)].min())
            self.log('gir_time_mean', self.gir_comp_time[np.where(self.gir_comp_time!=0)].mean())
            self.log('gir_time_max', self.gir_comp_time[np.where(self.gir_comp_time!=0)].max())
            self.log('cnn_time_min', self.cnn_comp_time[np.where(self.cnn_comp_time!=0)].min())
            self.log('cnn_time_mean', self.cnn_comp_time[np.where(self.cnn_comp_time!=0)].mean())
            self.log('cnn_time_max', self.cnn_comp_time[np.where(self.cnn_comp_time!=0)].max())
            self.log('don_time_min', self.don_comp_time[np.where(self.don_comp_time!=0)].min())
            self.log('don_time_mean', self.don_comp_time[np.where(self.don_comp_time!=0)].mean())
            self.log('don_time_max', self.don_comp_time[np.where(self.don_comp_time!=0)].max())
        ep = torch.arange(self.cnn_metrics.shape[0])
        plt.plot(ep, self.cnn_metrics.mean(-1), label='CNN')
        plt.fill_between(ep, self.cnn_metrics.mean(-1) - self.cnn_metrics.std(-1), self.cnn_metrics.mean(-1) + self.cnn_metrics.std(-1), alpha=0.2)
        plt.plot(ep, self.gir_metrics.mean(-1), label='Direct Girsanov')
        plt.fill_between(ep, self.gir_metrics.mean(-1) - self.gir_metrics.std(-1), self.gir_metrics.mean(-1) + self.gir_metrics.std(-1), alpha=0.2)
        plt.ylabel('Relative Error')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig('muBx_2d_cnn_gir.png')
        plt.clf()
        plt.plot(ep, self.cnn_metrics.mean(-1), label='CNN')
        plt.fill_between(ep, self.cnn_metrics.mean(-1) - self.cnn_metrics.std(-1), self.cnn_metrics.mean(-1) + self.cnn_metrics.std(-1), alpha=0.2)
        plt.ylabel('Relative Error')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig('muBx_2d_cnn.png')
        plt.clf()
        plt.plot(ep, self.em_comp_time.mean(-1), label='EM')
        plt.fill_between(ep, self.em_comp_time.mean(-1) - self.em_comp_time.std(-1), self.em_comp_time.mean(-1) + self.em_comp_time.std(-1), alpha=0.2)
        plt.plot(ep, self.gir_comp_time.mean(-1), label='Direct Girsanov')
        plt.fill_between(ep, self.gir_comp_time.mean(-1) - self.gir_comp_time.std(-1), self.gir_comp_time.mean(-1) + self.gir_comp_time.std(-1), alpha=0.2)
        plt.plot(ep, self.cnn_comp_time.mean(-1), label='CNN')
        plt.fill_between(ep, self.cnn_comp_time.mean(-1) - self.cnn_comp_time.std(-1), self.cnn_comp_time.mean(-1) + self.cnn_comp_time.std(-1), alpha=0.2)
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
    gir_loss_min = []
    gir_loss_mean = []
    gir_loss_max = []
    cnn_loss_min = []
    cnn_loss_mean = []
    cnn_loss_max = []
    don_loss_min = []
    don_loss_mean = []
    don_loss_max = []
    gir_time_min = []
    gir_time_mean = []
    gir_time_max = []
    cnn_time_min = []
    cnn_time_mean = []
    cnn_time_max = []
    don_time_min = []
    don_time_mean = []
    don_time_max = []
    em_time_min = []
    em_time_mean = []
    em_time_max = []
    
    for i in range(1,21):
        X = 0.5
        T = i * 0.05
        num_time = 25 * i
        dim = 10
        num_samples = 220
        batch_size = 10
        N = 500
        xs = torch.rand(num_samples,dim) * X
        ts = torch.rand(num_samples,1) * T
        dataset = torch.cat((xs,ts),dim=1)
        data_train = dataset[:20,:]
        data_val = dataset[20:,:]
        
        train_kwargs = {'batch_size': batch_size,
                'shuffle': True,
                'num_workers': 1}

        test_kwargs = {'batch_size': batch_size,
                'shuffle': False,
                'num_workers': 1}

        n_batch_val = int(num_samples / batch_size)-1

        train_loader = torch.utils.data.DataLoader(data_train,**train_kwargs)
        val_loader = torch.utils.data.DataLoader(data_val, **test_kwargs)

        model = FKModule(X=X, T=T, batch_size=batch_size, dim=dim, num_time=num_time, N=N, n_batch_val=n_batch_val)
        trainer = pl.Trainer(max_epochs=1, gpus=1, check_val_every_n_epoch=1)
        trainer.fit(model, train_loader, val_loader)
        
        gir_loss_min.append(trainer.logged_metrics['gir_loss_min'].item())
        gir_loss_mean.append(trainer.logged_metrics['gir_loss_mean'].item())
        gir_loss_max.append(trainer.logged_metrics['gir_loss_max'].item())
        cnn_loss_min.append(trainer.logged_metrics['cnn_loss_min'].item())
        cnn_loss_mean.append(trainer.logged_metrics['cnn_loss_mean'].item())
        cnn_loss_max.append(trainer.logged_metrics['cnn_loss_max'].item())
        don_loss_min.append(trainer.logged_metrics['don_loss_min'].item())
        don_loss_mean.append(trainer.logged_metrics['don_loss_mean'].item())
        don_loss_max.append(trainer.logged_metrics['don_loss_max'].item())
        
        gir_time_min.append(trainer.logged_metrics['gir_time_min'].item())
        gir_time_mean.append(trainer.logged_metrics['gir_time_mean'].item())
        gir_time_max.append(trainer.logged_metrics['gir_time_max'].item())
        cnn_time_min.append(trainer.logged_metrics['cnn_time_min'].item())
        cnn_time_mean.append(trainer.logged_metrics['cnn_time_mean'].item())
        cnn_time_max.append(trainer.logged_metrics['cnn_time_max'].item())
        don_time_min.append(trainer.logged_metrics['don_time_min'].item())
        don_time_mean.append(trainer.logged_metrics['don_time_mean'].item())
        don_time_max.append(trainer.logged_metrics['don_time_max'].item())
        em_time_min.append(trainer.logged_metrics['em_time_min'].item())
        em_time_mean.append(trainer.logged_metrics['em_time_mean'].item())
        em_time_max.append(trainer.logged_metrics['em_time_max'].item())
        #print(trainer.logged_metrics['val_loss'])
        #print(trainer.logged_metrics['train_loss'])
    #ep = torch.arange(18)
    with open('/scratch/xx84/girsanov/pde_rnn/cnn_loss_mean.npy', 'wb') as f:
        np.save(f, np.array(cnn_loss_mean))
    with open('/scratch/xx84/girsanov/pde_rnn/cnn_loss_min.npy', 'wb') as f:
        np.save(f, np.array(cnn_loss_min))
    with open('/scratch/xx84/girsanov/pde_rnn/cnn_loss_max.npy', 'wb') as f:
        np.save(f, np.array(cnn_loss_max))
    with open('/scratch/xx84/girsanov/pde_rnn/gir_loss_mean.npy', 'wb') as f:
        np.save(f, np.array(gir_loss_mean))
    with open('/scratch/xx84/girsanov/pde_rnn/gir_loss_min.npy', 'wb') as f:
        np.save(f, np.array(gir_loss_min))
    with open('/scratch/xx84/girsanov/pde_rnn/gir_loss_max.npy', 'wb') as f:
        np.save(f, np.array(gir_loss_max))
    with open('/scratch/xx84/girsanov/pde_rnn/don_loss_mean.npy', 'wb') as f:
        np.save(f, np.array(don_loss_mean))
    with open('/scratch/xx84/girsanov/pde_rnn/don_loss_min.npy', 'wb') as f:
        np.save(f, np.array(don_loss_min))
    with open('/scratch/xx84/girsanov/pde_rnn/don_loss_max.npy', 'wb') as f:
        np.save(f, np.array(don_loss_max))
    
    with open('/scratch/xx84/girsanov/pde_rnn/cnn_time_mean.npy', 'wb') as f:
        np.save(f, np.array(cnn_time_mean))
    with open('/scratch/xx84/girsanov/pde_rnn/cnn_time_min.npy', 'wb') as f:
        np.save(f, np.array(cnn_time_min))
    with open('/scratch/xx84/girsanov/pde_rnn/cnn_time_max.npy', 'wb') as f:
        np.save(f, np.array(cnn_time_max))
    with open('/scratch/xx84/girsanov/pde_rnn/gir_time_mean.npy', 'wb') as f:
        np.save(f, np.array(gir_time_mean))
    with open('/scratch/xx84/girsanov/pde_rnn/gir_time_min.npy', 'wb') as f:
        np.save(f, np.array(gir_time_min))
    with open('/scratch/xx84/girsanov/pde_rnn/gir_time_max.npy', 'wb') as f:
        np.save(f, np.array(gir_time_max))
    with open('/scratch/xx84/girsanov/pde_rnn/don_time_mean.npy', 'wb') as f:
        np.save(f, np.array(don_time_mean))
    with open('/scratch/xx84/girsanov/pde_rnn/don_time_min.npy', 'wb') as f:
        np.save(f, np.array(don_time_min))
    with open('/scratch/xx84/girsanov/pde_rnn/don_time_max.npy', 'wb') as f:
        np.save(f, np.array(don_time_max))
    with open('/scratch/xx84/girsanov/pde_rnn/em_time_mean.npy', 'wb') as f:
        np.save(f, np.array(em_time_mean))
    with open('/scratch/xx84/girsanov/pde_rnn/em_time_min.npy', 'wb') as f:
        np.save(f, np.array(em_time_min))
    with open('/scratch/xx84/girsanov/pde_rnn/em_time_max.npy', 'wb') as f:
        np.save(f, np.array(em_time_max))
    """
    plt.ylim(0, np.array(don_max)+1)
    plt.plot(ep, np.array(gir_mean), label='Direct Girsanov')
    plt.fill_between(ep, np.array(gir_min), np.array(gir_max), alpha=0.2)
    plt.ylim(0, np.array(don_max)+1)
    plt.plot(ep, np.array(cnn_mean), label='CNN')
    plt.fill_between(ep, np.array(cnn_min), np.array(cnn_max), alpha=0.2)
    plt.ylim(0, np.array(don_max)+1)
    plt.plot(ep, np.array(don_mean), label='DeepONet')
    plt.fill_between(ep, np.array(don_min), np.array(don_max), alpha=0.2)
    plt.ylim(0, np.array(don_max)+1)
    plt.ylabel('Loss')
    plt.xlabel('Terminal Time')
    plt.legend()
    plt.savefig('loss_cnn.png')
    plt.clf()
"""