"""
This code is designed to train ngo on high-dimensional setting, the difference between this and fbsde_ngo.py is that
this one's 
"""
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

def b(t,x, coef, dim):
    x = x.unsqueeze(-1)
    #x = x/dim
    x0 = (x ** 0)/dim
    x1 = (torch.sin(x))/dim
    x2 = torch.cos(x)/dim
    vals = torch.cat((x0,x1,x2),axis=-1)
    return (coef * vals).sum(-1)

def sigma(t,x):
    return torch.ones_like(t)#torch.exp(-t)

def g(x):
    return torch.sin(2*np.pi*x).sum(-1)

def h(t,x,y,z,coef):
    x0 = torch.sin(x).sum(-1)
    x1 = (z ** 2).sum(-1)
    x2 = torch.cos(t+y)
    vals = torch.cat((x0.unsqueeze(-1),x1.unsqueeze(-1),x2.unsqueeze(-1)),axis=-1)
    return (coef * vals).sum(-1)

class CNN_expmart(nn.Module):
    def __init__(self, input_size, hidden_size, num_outputs):
        super(CNN_expmart, self).__init__()
        #self.num_layers = num_layers
        #self.hidden_size = hidden_size
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=1, padding=0)
        self.act1 = nn.Softplus()
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, padding=0)
        self.act2 = nn.Softplus()
        self.conv3 = nn.Conv1d(in_channels=hidden_size, out_channels=num_outputs, kernel_size=1, padding=0)
        #self.act3 = nn.Softplus()
        #self.conv4 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, padding=0)
        #self.act4 = nn.Softplus()
        #self.conv5 = nn.Conv1d(in_channels=hidden_size, out_channels=num_outputs, kernel_size=1, padding=0)
	
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        #out = self.act3(out)
        #out = self.conv4(out)
        #out = self.act4(out)
        #out = self.conv5(out)
        return out

class CNN_zt(nn.Module):
    def __init__(self, input_size, hidden_size, num_outputs):
        super(CNN_zt, self).__init__()
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
    def __init__(self, N = 2000, lr = 1e-3, X = 1., t0=0., T = 0.1, dim = 2, batch_size = 100, num_time = 100, n_batch_val=100):
        super().__init__()
        # define normalizing flow to model the conditional distribution rho(x,t)=p(y|x,t)
        self.num_time = num_time
        self.T = torch.Tensor([T]).to(device)
        self.t0 = torch.Tensor([t0]).to(device)
        self.t = torch.linspace(self.t0.item(),self.T.item(),steps=self.num_time).to(device)
        self.dim = dim
        self.batch_size = batch_size
        # input size is dimension of brownian motion x 2, since the input to the RNN block is W_s^x and dW_s^x
        input_size = self.dim * 2 + 1
        # hidden_size is dimension of the RNN output
        hidden_size = 50 + self.dim * 5
        # num_outputs is the number of ln(rho(x,t))
        num_outputs = self.dim
        self.expmart_cnn = CNN_expmart(input_size, hidden_size, num_outputs)
        self.zt_cnn = CNN_zt(input_size=dim+1, hidden_size=50 + self.dim * 5, num_outputs=self.dim)
        #self.sequence.load_state_dict(torch.load('/scratch/xx84/girsanov/pde_rnn/rnn_prior.pt'))

        # define the learning rate
        self.lr = lr
                
        # define number of paths used and grid of PDE
        self.N = N
        self.dt = self.t[1]-self.t[0] # define time step

        # define the brwonian motion starting at zero
        #dB = torch.sqrt(self.dt) * torch.randn(self.num_time-1, self.N, self.batch_size, self.dim).to(device)
        #zeros = torch.zeros(1, self.N, self.batch_size, self.dim).to(device)
        #zeros.requires_grad = True
        #self.dB = torch.cat((zeros, dB),dim=0)
        self.dB = torch.sqrt(self.dt) * torch.randn(self.num_time, self.N, self.batch_size, self.dim).to(device)
        self.dB[0,:,:,:] = 0
        self.B0 = self.dB
        self.B0 = torch.cumsum(self.B0, dim=0)
        
        
        self.metrics = torch.zeros((50,n_batch_val))
        self.gir_metrics = torch.zeros((50,n_batch_val))
        self.comp_time = torch.zeros((50,n_batch_val))
        self.gir_comp_time = torch.zeros((50,n_batch_val))
        self.cnn_comp_time = torch.zeros((50,n_batch_val))
        self.epochs = torch.linspace(0,49,50)
        
        self.relu = torch.nn.Softplus()

        self.coef_train = torch.rand(10,1,1,3).to(device)
        
        self.relu = torch.nn.Softplus()

    def loss(self, xt, coef, coef1, em = False):
       # calculation with girsanov
        xs = xt[:,:-1]
        ts = xt[:,-1]
        sigmas = sigma(self.t[0],xs)
        xi = torch.cumsum(self.dB * sigmas.unsqueeze(0).unsqueeze(0),dim=0) + xs.unsqueeze(0).unsqueeze(0).repeat(self.num_time,N,1,1)
        xi.requires_grad = True
        sigmas = sigma(self.t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), xi)
        mu = b(self.t, xi, coef, self.dim)/sigmas
        start = time.time()
        mart = torch.cumsum(mu * self.dB, dim=-1) - 0.5 * torch.cumsum(mu ** 2, dim=-1) * self.dt
        expmart = torch.exp(mart.sum(-1))

        xT = xi[-1,:,:,:]
        yT = g(xT) * expmart[-1,:,:]
        yi = torch.zeros(self.num_time, self.N, self.batch_size).to(device)
        yi[-1,:,:] = yT
        vi = yT.mean(0)
        zi_gir = torch.zeros_like(xi)
        z_current = sigma(torch.Tensor([T]),xT).to(device) * torch.autograd.grad(outputs=vi,inputs=xT,grad_outputs=torch.ones_like(vi).to(device),retain_graph=True)[0]
        zi_gir[-1,:,:,:] = z_current
        for i in reversed(range(1,self.num_time)):
            x_current = xi[i,:,:,:]
            t_current = self.t[i]
            yi[i-1,:,:] = yi[i,:,:] + h(t_current,x_current,yi[i,:,:],z_current,coef1) * self.dt * expmart[i-1,:,:]
            vi = yi[i-1,:,:].mean(0)
            z_current = sigma(t_current,x_current).to(device) * torch.autograd.grad(outputs=vi,inputs=x_current,grad_outputs=torch.ones_like(vi).to(device),retain_graph=True)[0]
            zi_gir[i-1,:,:,:] = z_current
            
        v_gir = yi.mean(1)
        end = time.time()
        time_gir = (end - start)
        # calculation with cnn
        muBx = b(self.t, xi, coef, self.dim)
        input = torch.zeros(self.num_time, self.N, self.batch_size, self.dim * 2 + 1).to(device)
        input[:muBx.shape[0],:,:,:] = torch.cat((muBx,self.dB,self.dt*torch.ones(self.dB.shape[0],self.dB.shape[1],self.dB.shape[2],1).to(device)),dim=-1)
        input_reshaped = input.reshape(input.shape[1]*input.shape[2], input.shape[3], input.shape[0])
        cnn_expmart = self.relu(self.expmart_cnn(input_reshaped).sum(-2)).reshape(self.num_time, self.N, self.batch_size)
        
        
        xT = xi[-1,:,:,:]
        yT = g(xT) * cnn_expmart[-1,:,:]
        
        #yT_cnn = (g(xi) * cnn_expmart).mean(1)
        
        yi = torch.zeros(self.num_time, self.N, self.batch_size).to(device)
        yi[-1,:,:] = yT
        vi = yi[-1,:,:].mean(0)
        zi_cnn = torch.zeros_like(xi)
        input_zi = torch.cat((yT.unsqueeze(-1),xT),dim=-1)
        input_zi = input_zi.reshape(input_zi.shape[0] * input_zi.shape[1], input_zi.shape[2], 1)
        z_current = self.zt_cnn(input_zi).reshape(xT.shape)
        #z_current = sigma(torch.Tensor([T]),xT).to(device) * torch.autograd.grad(outputs=vi,inputs=xT,grad_outputs=torch.ones_like(vi).to(device),retain_graph=True)[0]
        zi_cnn[-1,:,:,:] = z_current
        for i in reversed(range(1,self.num_time)):
            x_current = xi[i,:,:,:]
            t_current = self.t[i]
            yi[i-1,:,:] = yi[i,:,:] + h(t_current,x_current,yi[i,:,:],z_current,coef1) * self.dt * cnn_expmart[i-1,:,:]
            vi = yi[i-1,:,:].mean(0)
            input_zi = torch.cat((yi[i-1,:,:].unsqueeze(-1),x_current),dim=-1)
            input_zi = input_zi.reshape(input_zi.shape[0] * input_zi.shape[1], input_zi.shape[2], 1)
            z_current = sigma(t_current,x_current).to(device) * self.zt_cnn(input_zi).reshape(xT.shape)
            #z_current = sigma(t_current,x_current).to(device) * torch.autograd.grad(outputs=vi,inputs=x_current,grad_outputs=torch.ones_like(vi).to(device),retain_graph=True)[0]
            zi_cnn[i-1,:,:,:] = z_current
            
        v_cnn = yi.mean(1)
        end = time.time()
        time_cnn = (end - start)
        if em:
            # calculation with EM
            xs = xt[:,:-1]
            ts = xt[:,-1]
            xi= xs.unsqueeze(0).repeat(N,1,1)
            xi = xi.unsqueeze(0)
            xi.requires_grad = True
            for i in range(0,self.num_time-1):
                x_current = xi[i,:,:,:] + b(self.t[i], xi[i,:,:,:], coef, self.dim) * self.dt + sigma(self.t[i], xi[i,:,:,:]).to(device) * self.dB[i,:,:,:]
                xi = torch.cat((xi, x_current.unsqueeze(0)),dim=0)
            
            xT = xi[-1,:,:,:]
            yT = g(xT)
            yi = torch.zeros(self.num_time, self.N, self.batch_size).to(device)
            yi[-1,:,:] = yT
            vi = yT.mean(0)
            zi_em = torch.zeros_like(xi)
            z_current = sigma(torch.Tensor([T]),xT).to(device) * torch.autograd.grad(outputs=vi,inputs=xT,grad_outputs=torch.ones_like(vi).to(device),retain_graph=True)[0]
            zi_em[-1,:,:,:] = z_current
            for i in reversed(range(1,self.num_time)):
                x_current = xi[i,:,:,:]
                t_current = self.t[i]
                yi[i-1,:,:] = yi[i,:,:] + h(t_current,x_current,yi[i,:,:],z_current, coef1) * self.dt
                vi = yi[i-1,:,:].mean(0)
                z_current = sigma(t_current,x_current).to(device) * torch.autograd.grad(outputs=vi,inputs=x_current,grad_outputs=torch.ones_like(vi).to(device),retain_graph=True)[0]
                zi_em[i-1,:,:,:] = z_current
                
            v_em = yi.mean(1)
            end = time.time()
            time_em = (end - start)
            return v_gir, v_cnn, v_em, time_gir, time_cnn, time_em, zi_em, zi_cnn, zi_gir
        return v_gir, v_cnn, time_gir, time_cnn, zi_cnn, zi_gir

    def training_step(self, batch, batch_idx):
        # REQUIRED
        xt = batch.to(device)
        v_gir, v_cnn, v_em, time_gir, time_cnn, time_em, zi_em, zi_cnn, zi_gir = self.loss(xt, coef=torch.rand(1,1,1,3).to(device), coef1=torch.rand(1,1,1,3).to(device), em=True)
        
        #v_cnn = v_cnn[~torch.any(v_cnn.isnan(),dim=1)]
        #v_cnn = v_cnn[~torch.any(v_cnn.isnan(),dim=1)]
        loss = F.l1_loss(v_cnn, v_gir)+F.l1_loss(zi_cnn, zi_gir)#+F.l1_loss(zi_cnn, zi_em)+F.l1_loss(v_cnn, v_em)#/(torch.abs(u_gir).mean())
        #tensorboard_logs = {'train_loss': loss_prior}
        self.log('train_loss', loss)
        #print(loss_total)
        return {'loss': loss}
        
        
    def validation_step(self, batch, batch_idx):
        #super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)
        xt = batch.to(device)
        v_cnn, v_gir, v_em, time_gir, time_cnn, time_em, zi_em, zi_cnn, zi_gir = self.loss(xt, coef=torch.rand(1,1,1,3).to(device), coef1=torch.rand(1,1,1,3).to(device), em=True)
        loss = F.mse_loss(v_cnn,v_em,reduction='mean')/(torch.abs(v_em).mean())
        loss_g = F.mse_loss(v_gir,v_em,reduction='mean')/(torch.abs(v_em).mean())
        print('Validation: {:.4f}, {:.4f}'.format(loss, loss_g))
        self.log('val_loss', loss)
        if not loss.isnan():
            self.metrics[self.current_epoch, batch_idx] = loss.item()
            self.gir_metrics[self.current_epoch, batch_idx] = loss_g.item()
            self.comp_time[self.current_epoch, batch_idx] = time_em
            self.gir_comp_time[self.current_epoch, batch_idx] = time_gir
            self.cnn_comp_time[self.current_epoch, batch_idx] = time_cnn
        ep = torch.arange(self.metrics.shape[0])
        plt.plot(ep, self.metrics.mean(-1), label='CNN')
        plt.fill_between(ep, self.metrics.mean(-1) - self.metrics.std(-1), self.metrics.mean(-1) + self.metrics.std(-1), alpha=0.2)
        plt.plot(ep, self.gir_metrics.mean(-1), label='Direct Girsanov')
        plt.fill_between(ep, self.gir_metrics.mean(-1) - self.gir_metrics.std(-1), self.gir_metrics.mean(-1) + self.gir_metrics.std(-1), alpha=0.2)
        plt.ylabel('Relative Error')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig('/scratch/xx84/girsanov/fbsde/ablation/figure/ngo_train_girloss_full_'+str(self.dim)+'.png')
        plt.clf()
        plt.plot(ep, self.metrics.mean(-1), label='CNN')
        plt.fill_between(ep, self.metrics.mean(-1) - self.metrics.std(-1), self.metrics.mean(-1) + self.metrics.std(-1), alpha=0.2)
        plt.ylabel('Relative Error')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig('/scratch/xx84/girsanov/fbsde/ablation/figure/ngo_train_girloss_ngo_'+str(self.dim)+'.png')
        plt.clf()
        plt.plot(ep, self.comp_time.mean(-1), label='EM')
        plt.fill_between(ep, self.comp_time.mean(-1) - self.comp_time.std(-1), self.comp_time.mean(-1) + self.comp_time.std(-1), alpha=0.2)
        plt.plot(ep, self.gir_comp_time.mean(-1), label='Direct Girsanov')
        plt.fill_between(ep, self.gir_comp_time.mean(-1) - self.gir_comp_time.std(-1), self.gir_comp_time.mean(-1) + self.gir_comp_time.std(-1), alpha=0.2)
        plt.plot(ep, self.cnn_comp_time.mean(-1), label='CNN')
        plt.fill_between(ep, self.cnn_comp_time.mean(-1) - self.cnn_comp_time.std(-1), self.cnn_comp_time.mean(-1) + self.cnn_comp_time.std(-1), alpha=0.2)
        plt.ylabel('Computation Time')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig('/scratch/xx84/girsanov/fbsde/ablation/figure/ngo_train_comptime_'+str(self.dim)+'.png')
        plt.clf()
        torch.save(self.expmart_cnn.state_dict(), '/scratch/xx84/girsanov/fbsde/ablation/trained_model/exp_cnn_'+str(self.dim)+'.pt')
        torch.save(self.zt_cnn.state_dict(), '/scratch/xx84/girsanov/fbsde/ablation/trained_model/zt_cnn_'+str(self.dim)+'.pt')
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
    
    for dim in range(11,20):
        x0 = 0.1
        X = 0.5
        T = 0.1
        t0 = 0.
        num_time = 40
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

        model = FKModule(X=X, t0=t0, T=T, batch_size=batch_size, dim=dim, num_time=num_time, N=N, n_batch_val=n_batch_val)
        trainer = pl.Trainer(max_epochs=10, gpus=1, check_val_every_n_epoch=1)
        trainer.fit(model, train_loader, val_loader)
        
        print(trainer.logged_metrics['val_loss'])
        print(trainer.logged_metrics['train_loss'])
