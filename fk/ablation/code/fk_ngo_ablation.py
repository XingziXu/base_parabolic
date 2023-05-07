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
import time

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

class CNN_expmart(nn.Module):
    def __init__(self, input_size, hidden_size, num_outputs):
        super(CNN_expmart, self).__init__()
        #self.num_layers = num_layers
        #self.hidden_size = hidden_size
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=1, padding=0)
        self.act1 = nn.Softplus()
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, padding=0)
        self.act2 = nn.Softplus()
        self.conv3 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, padding=0)
        self.act3 = nn.Softplus()
        self.conv4 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, padding=0)
        self.act4 = nn.Softplus()
        self.conv5 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, padding=0)
        self.act5 = nn.Softplus()
        self.conv6 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, padding=0)
        self.act6 = nn.Softplus()
        self.conv7 = nn.Conv1d(in_channels=hidden_size, out_channels=num_outputs, kernel_size=1, padding=0)
	
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.conv5(out)
        out = self.act5(out)
        out = self.conv6(out)
        out = self.act6(out)
        out = self.conv7(out)
        return out


class FKModule(pl.LightningModule):
    def __init__(self, N = 2000, lr = 1e-3, X = 1., T = 0.1, dim = 2, batch_size = 100, num_time = 100, n_batch_val=100):
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
        hidden_size = 50 + self.dim * 5
        # num_outputs is the number of ln(rho(x,t))
        num_outputs = self.dim
        self.expmart_cnn = CNN_expmart(input_size, hidden_size, num_outputs)
        #self.sequence.load_state_dict(torch.load('/scratch/xx84/girsanov/pde_rnn/rnn_prior.pt'))

        # define the learning rate
        self.lr = lr
                
        # define number of paths used and grid of PDE
        self.N = N
        self.dt = self.t[1]-self.t[0] # define time step

        # define the brwonian motion starting at zero
        self.dB = np.sqrt(self.dt.item()) * np.random.randn(self.t.shape[0], self.N, self.batch_size, self.dim)
        self.dB[0,:,:,:] = 0 
        self.B0 = self.dB.copy()
        self.B0 = torch.Tensor(self.B0.cumsum(0)).to(device)
        self.dB = torch.Tensor(self.dB).to(device)
        
        self.cnn_metrics = torch.zeros((50,n_batch_val))
        self.gir_metrics = torch.zeros((50,n_batch_val))
        self.em_comp_time = torch.zeros((50,n_batch_val))
        self.gir_comp_time = torch.zeros((50,n_batch_val))
        self.cnn_comp_time = torch.zeros((50,n_batch_val))
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
        start = time.time()
        x = torch.zeros(self.num_time, self.N, batch_size, self.dim).to(device)
        x[0,:,:,:] = xs.squeeze()
        for i in range(self.num_time-1):
            x[i+1,:,:,:] = x[i,:,:,:] + drift(x[i,:,:,:], coef, self.dim).squeeze() * self.dt + self.dB[i,:,:,:]
        p0mux = initial(x)
        u_em = p0mux.mean(1)
        end = time.time()
        time_em = (end - start)
        # calculate values using girsanov
        start = time.time()
        muBx = drift(Bx, coef, self.dim)
        expmart = torch.exp((torch.cumsum(muBx*self.dB,dim=0) - 0.5 * torch.cumsum((muBx ** 2) * self.dt,dim=0)).sum(-1))
        u_gir = (p0Bx*expmart).mean(1)
        end = time.time()
        time_gir = (end - start)
        # calculate values using RNN
        start = time.time()
        input = torch.zeros(self.num_time, self.N, self.batch_size, self.dim * 2 + 1).to(device)
        input[:muBx.shape[0],:,:,:] = torch.cat((muBx,self.dB,self.dt*torch.ones(self.dB.shape[0],self.dB.shape[1],self.dB.shape[2],1).to(device)),dim=-1)
        input_reshaped = input.reshape(input.shape[1]*input.shape[2], input.shape[3], input.shape[0])
        cnn_expmart = self.relu(self.expmart_cnn(input_reshaped).sum(-2)).reshape(p0Bx.shape)
        u_cnn = (p0Bx*cnn_expmart).mean(1)
        end = time.time()
        time_cnn = (end - start)
        return u_em, u_gir, u_cnn, time_em, time_gir, time_cnn

    def training_step(self, batch, batch_idx):
        # REQUIRED
        xt = batch.to(device)
        u_em, u_gir, u_cnn, time_em, time_gir, time_cnn = self.loss(xt, coef=torch.rand(1,1,1,3).to(device))
        loss = F.l1_loss(u_cnn, u_gir)#/(torch.abs(u_gir).mean())
        #tensorboard_logs = {'train_loss': loss_prior}
        self.log('train_loss', loss)
        #print(loss_total)
        return {'loss': loss}
        
        
    def validation_step(self, batch, batch_idx):
        xt = batch.to(device)
        u_em, u_gir, u_cnn, time_em, time_gir, time_cnn = self.loss(xt, coef=torch.rand(1,1,1,3).to(device))
        loss_cnn = F.mse_loss(u_cnn,u_em,reduction='mean')/(torch.abs(u_em).mean())
        loss_gir = F.mse_loss(u_gir,u_em,reduction='mean')/(torch.abs(u_em).mean())
        print('Validation: {:.4f}, {:.4f}'.format(loss_cnn, loss_gir))
        self.log('val_loss', loss_cnn)
        if not loss_cnn.isnan():
            self.cnn_metrics[self.current_epoch, batch_idx] = loss_cnn.item()
            self.gir_metrics[self.current_epoch, batch_idx] = loss_gir.item()
            self.em_comp_time[self.current_epoch, batch_idx] = time_em
            self.gir_comp_time[self.current_epoch, batch_idx] = time_gir
            self.cnn_comp_time[self.current_epoch, batch_idx] = time_cnn
        ep = torch.arange(self.cnn_metrics.shape[0])
        plt.plot(ep, self.cnn_metrics.mean(-1), label='CNN')
        plt.fill_between(ep, self.cnn_metrics.mean(-1) - self.cnn_metrics.std(-1), self.cnn_metrics.mean(-1) + self.cnn_metrics.std(-1), alpha=0.2)
        plt.plot(ep, self.gir_metrics.mean(-1), label='Direct Girsanov')
        plt.fill_between(ep, self.gir_metrics.mean(-1) - self.gir_metrics.std(-1), self.gir_metrics.mean(-1) + self.gir_metrics.std(-1), alpha=0.2)
        plt.ylabel('Relative Error')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig('/scratch/xx84/girsanov/fk/high_dim/figure/ngo_train_girloss_full_'+str(self.dim)+'.png')
        plt.clf()
        plt.plot(ep, self.cnn_metrics.mean(-1), label='CNN')
        plt.fill_between(ep, self.cnn_metrics.mean(-1) - self.cnn_metrics.std(-1), self.cnn_metrics.mean(-1) + self.cnn_metrics.std(-1), alpha=0.2)
        plt.ylabel('Relative Error')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig('/scratch/xx84/girsanov/fk/high_dim/figure/ngo_train_girloss_ngo_'+str(self.dim)+'.png')
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
        plt.savefig('/scratch/xx84/girsanov/fk/high_dim/figure/ngo_train_comptime_'+str(self.dim)+'.png')
        plt.clf()
        torch.save(self.expmart_cnn.state_dict(), '/scratch/xx84/girsanov/fk/high_dim/trained_model/ngo_'+str(self.dim)+'.pt')
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
    
    x0 = 0.1
    X = 0.5
    T = 0.1
    num_time = 40
    dim = 1
    num_samples = 12000
    batch_size = 10
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

    model = FKModule(X=X, T=T, batch_size=batch_size, dim=dim, num_time=num_time, N=N, n_batch_val=n_batch_val)
    trainer = pl.Trainer(max_epochs=50, gpus=1, check_val_every_n_epoch=1)
    trainer.fit(model, train_loader, val_loader)
    
    print(trainer.logged_metrics['val_loss'])
    print(trainer.logged_metrics['train_loss'])
