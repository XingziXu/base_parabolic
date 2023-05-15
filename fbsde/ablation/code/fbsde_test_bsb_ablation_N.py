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


def b(t,x, coef):
    mu = torch.zeros_like(x)
    mu.requires_grad = True
    return mu

def sigma(t,x):
    return torch.sqrt(torch.Tensor([2.])) * torch.ones_like(t)#torch.exp(-t)

def g(x):
    return (x ** 2).sum(-1)

def h(t,x,y,z,coef,r,sigma):
    return r*(y-(z*x).sum(-1))

def sigma_back(x,sigma,z):
    return sigma * (x*z).sum(-1)

def exact(T,t,r,sigma,x):
    return torch.exp((r+sigma**2)*(T-t)).squeeze()*g(x)

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
    def __init__(self, N = 2000, lr = 1e-3, X = 1., t0=0., T = 0.1, dim = 2, batch_size = 100, num_time = 100, n_batch_val=100, m=100, p=15):
        super().__init__()
        # define normalizing flow to model the conditional distribution rho(x,t)=p(y|x,t)
        self.m = m
        self.p = p
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

        self.X = X
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
        
        
        self.em_metrics = torch.zeros((50,n_batch_val))
        self.cnn_metrics = torch.zeros((50,n_batch_val))
        self.gir_metrics = torch.zeros((50,n_batch_val))
        self.don_metrics = torch.zeros((50,n_batch_val))
        self.em_comp_time = torch.zeros((50,n_batch_val))
        self.cnn_comp_time = torch.zeros((50,n_batch_val))
        self.gir_comp_time = torch.zeros((50,n_batch_val))
        self.don_comp_time = torch.zeros((50,n_batch_val))
        self.epochs = torch.linspace(0,49,50)
        
        self.relu = torch.nn.Softplus()

        self.coef_train = torch.rand(10,1,1,3).to(device)
        
        self.relu = torch.nn.Softplus()
        
        self.branch = MLP(input_dim=self.m, hidden_dim=70+5*dim, output_dim=self.p) # branch network
        self.trunk = MLP(input_dim=dim+1, hidden_dim=50+5*dim, output_dim=self.p) # trunk network
        self.sensors = g((torch.linspace(self.t0.item(), self.T.item(), self.m).unsqueeze(-1).repeat(1,self.dim) * self.X).to(device))

    def loss(self, xt, coef, coef1, em = False):
        # calculation with cnn
        xs = xt[:,:-1]
        ts = xt[:,-1]
        sigmas = 0.1
        r = 0.05
        xi = xs.unsqueeze(0).unsqueeze(0) * torch.exp(torch.cumsum(self.dB * sigmas,dim=0) - (sigmas ** 2) * self.t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
        xi.requires_grad = True
        muBx = b(self.t, xi, coef)
        input = torch.zeros(self.num_time, self.N, self.batch_size, self.dim * 2 + 1).to(device)
        input[:muBx.shape[0],:,:,:] = torch.cat((muBx,self.dB,self.dt*torch.ones(self.dB.shape[0],self.dB.shape[1],self.dB.shape[2],1).to(device)),dim=-1)
        input_reshaped = input.reshape(input.shape[1]*input.shape[2], input.shape[3], input.shape[0])
        start = time.time()
        cnn_expmart = self.relu(self.expmart_cnn(input_reshaped).sum(-2)).reshape(self.num_time, self.N, self.batch_size)
        xT = xi[-1,:,:,:]
        yT = g(xT) * cnn_expmart[-1,:,:]
        yi = torch.zeros(self.num_time, self.N, self.batch_size).to(device)
        yi[-1,:,:] = yT
        zi_cnn = torch.zeros_like(xi)
        input_zi = torch.cat((g(xi).unsqueeze(-1),xi),dim=-1)
        input_zi = input_zi.reshape(input_zi.shape[1] * input_zi.shape[2], input_zi.shape[3], input_zi.shape[0])
        zi_cnn = self.zt_cnn(input_zi).reshape(xi.shape)
        hi = h(self.t.unsqueeze(-1).unsqueeze(-1),xi,yi,zi_cnn,coef1,r,sigmas) * self.dt * cnn_expmart
        hi = torch.flip(hi,dims=[0])
        yi_cumsum = yT + torch.cumsum(hi,dim=0)
        v_cnn = torch.flip(yi_cumsum,dims=[0]).mean(1)
        end = time.time()
        time_cnn = (end - start)
        # calculation with don
        branchs = self.branch(self.sensors.unsqueeze(0)).repeat(self.batch_size, 1)
        v_don = torch.zeros(self.num_time, self.batch_size).to(device)
        start = time.time()
        for i in range(self.num_time-1):
            trunks = self.trunk(torch.cat((xs,i*self.dt*torch.ones(xs.shape[0],1).to(device)),dim=1))
            v_don[i,:] = (branchs * trunks).sum(1)
        v_don = v_don
        end = time.time()
        time_don = (end - start)
        if em:
            # calculation with EM
            yi = torch.zeros(self.num_time, self.N, self.batch_size).to(device)
            zi_em = torch.zeros_like(self.dB)
            start = time.time()
            xi = xs.unsqueeze(0).unsqueeze(0) * torch.exp(torch.cumsum(self.dB * sigmas,dim=0) - (sigmas ** 2) * self.t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
            xi.requires_grad = True
            xT = xi[-1,:,:,:]
            yT = g(xT)
            yi[-1,:,:] = yT
            vi = yT.mean(0)
            z_current = torch.autograd.grad(outputs=vi,inputs=xT,grad_outputs=torch.ones_like(vi).to(device),retain_graph=True)[0]
            zi_em[-1,:,:,:] = z_current
            for i in reversed(range(1,self.num_time)):
                x_current = xi[i,:,:,:]
                t_current = self.t[i]
                t_current.requires_grad = True
                yi[i-1,:,:] = yi[i,:,:] + h(t_current,x_current,yi[i,:,:],z_current,coef1,r,sigmas) * self.dt
                vi = yi[i-1,:,:].mean(0)
                z_current = torch.autograd.grad(outputs=vi,inputs=xT,grad_outputs=torch.ones_like(vi).to(device),retain_graph=True)[0]
                zi_em[i-1,:,:,:] = z_current
            v_em = yi.mean(1)
            end = time.time()
            time_em = (end - start)
            # calculate analytical solution
            v_gt = torch.zeros_like(v_cnn)
            for i in range(0,self.num_time):
                t_current = self.t[i].unsqueeze(0).unsqueeze(0).repeat(self.batch_size,1)
                v_gt[i,:] = exact(self.T,t_current,r,sigmas,xs)
                
                
            return v_cnn, v_don, v_em, v_gt, time_cnn, time_don, time_em
        return v_cnn, v_don, v_gt, time_cnn, time_don

    def training_step(self, batch, batch_idx):
        # REQUIRED
        xt = batch.to(device)
        v_cnn, v_don, v_em, v_gt, time_cnn, time_don, time_em = self.loss(xt, coef=torch.rand(1,1,1,3).to(device), coef1=torch.rand(1,1,1,3).to(device), em=True)
        
        #v_cnn = v_cnn[~torch.any(v_cnn.isnan(),dim=1)]
        #v_cnn = v_cnn[~torch.any(v_cnn.isnan(),dim=1)]
        loss = F.l1_loss(v_cnn, v_don)#+F.l1_loss(zi_cnn, zi_gir)#+F.l1_loss(zi_cnn, zi_em)+F.l1_loss(v_cnn, v_em)#/(torch.abs(u_gir).mean())
        #tensorboard_logs = {'train_loss': loss_prior}
        self.log('train_loss', loss)
        #print(loss_total)
        return {'loss': loss}
        
        
    def validation_step(self, batch, batch_idx):
        #super().on_validation_model_eval(*args, **kwargs)
        self.branch.load_state_dict(torch.load('/scratch/xx84/girsanov/fbsde/ablation/trained_model_N/branch_'+str(self.N)+'.pt'))
        self.trunk.load_state_dict(torch.load('/scratch/xx84/girsanov/fbsde/ablation/trained_model_N/trunk_'+str(self.N)+'.pt'))
        self.expmart_cnn.load_state_dict(torch.load('/scratch/xx84/girsanov/fbsde/ablation/trained_model_N/exp_cnn_'+str(self.N)+'.pt'))
        self.zt_cnn.load_state_dict(torch.load('/scratch/xx84/girsanov/fbsde/ablation/trained_model_N/zt_cnn_'+str(self.N)+'.pt'))
        torch.set_grad_enabled(True)
        xt = batch.to(device)
        v_cnn, v_don, v_gir, v_gt, time_cnn, time_don, time_em = self.loss(xt, coef=torch.rand(1,1,1,3).to(device), coef1=torch.rand(1,1,1,3).to(device), em=True)
        loss_cnn = F.mse_loss(v_cnn,v_gt,reduction='mean')/(torch.abs(v_gt).mean())
        loss_gir = F.mse_loss(v_gir,v_gt,reduction='mean')/(torch.abs(v_gt).mean())
        loss_don = F.mse_loss(v_don,v_gt,reduction='mean')/(torch.abs(v_gt).mean())
        print('Validation: {:.4f}, {:.4f}'.format(loss_cnn, loss_gir))
        self.log('val_loss', loss_cnn)
        if not loss_cnn.isnan():
            if not loss_cnn.isinf():
                if not loss_gir.isnan():
                    if not loss_gir.isinf():
                        self.cnn_metrics[self.current_epoch, batch_idx] = loss_cnn.item()
                        self.gir_metrics[self.current_epoch, batch_idx] = loss_gir.item()
                        self.don_metrics[self.current_epoch, batch_idx] = loss_don.item()
                        self.em_comp_time[self.current_epoch, batch_idx] = time_em
                        self.gir_comp_time[self.current_epoch, batch_idx] = time_em
                        self.cnn_comp_time[self.current_epoch, batch_idx] = time_cnn
                        self.don_comp_time[self.current_epoch, batch_idx] = time_don
                        self.log('gir_loss_min', self.gir_metrics[np.where(self.gir_metrics!=0)].min())
                        self.log('gir_loss_mean', self.gir_metrics[np.where(self.gir_metrics!=0)].mean())
                        self.log('gir_loss_max', self.gir_metrics[np.where(self.gir_metrics!=0)].max())
                        self.log('gir_loss_var', torch.var(self.gir_metrics[np.where(self.gir_metrics!=0)]))
                        self.log('cnn_loss_min', self.cnn_metrics[np.where(self.cnn_metrics!=0)].min())
                        self.log('cnn_loss_mean', self.cnn_metrics[np.where(self.cnn_metrics!=0)].mean())
                        self.log('cnn_loss_max', self.cnn_metrics[np.where(self.cnn_metrics!=0)].max())
                        self.log('cnn_loss_var', torch.var(self.cnn_metrics[np.where(self.cnn_metrics!=0)]))
                        self.log('don_loss_min', self.don_metrics[np.where(self.don_metrics!=0)].min())
                        self.log('don_loss_mean', self.don_metrics[np.where(self.don_metrics!=0)].mean())
                        self.log('don_loss_max', self.don_metrics[np.where(self.don_metrics!=0)].max())
                        self.log('don_loss_var', torch.var(self.don_metrics[np.where(self.don_metrics!=0)]))
                        
                        
                        self.log('em_time_min', self.em_comp_time[np.where(self.em_comp_time!=0)].min())
                        self.log('em_time_mean', self.em_comp_time[np.where(self.em_comp_time!=0)].mean())
                        self.log('em_time_max', self.em_comp_time[np.where(self.em_comp_time!=0)].max())
                        self.log('em_time_var', torch.var(self.em_comp_time[np.where(self.em_comp_time!=0)]))
                        self.log('gir_time_min', self.gir_comp_time[np.where(self.gir_comp_time!=0)].min())
                        self.log('gir_time_mean', self.gir_comp_time[np.where(self.gir_comp_time!=0)].mean())
                        self.log('gir_time_max', self.gir_comp_time[np.where(self.gir_comp_time!=0)].max())
                        self.log('gir_time_var', torch.var(self.gir_comp_time[np.where(self.gir_comp_time!=0)]))
                        self.log('cnn_time_min', self.cnn_comp_time[np.where(self.cnn_comp_time!=0)].min())
                        self.log('cnn_time_mean', self.cnn_comp_time[np.where(self.cnn_comp_time!=0)].mean())
                        self.log('cnn_time_max', self.cnn_comp_time[np.where(self.cnn_comp_time!=0)].max())
                        self.log('cnn_time_var', torch.var(self.cnn_comp_time[np.where(self.cnn_comp_time!=0)]))
                        self.log('don_time_min', self.don_comp_time[np.where(self.don_comp_time!=0)].min())
                        self.log('don_time_mean', self.don_comp_time[np.where(self.don_comp_time!=0)].mean())
                        self.log('don_time_max', self.don_comp_time[np.where(self.don_comp_time!=0)].max())
                        self.log('don_time_var', torch.var(self.don_comp_time[np.where(self.don_comp_time!=0)]))
        return

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
    gir_loss_var = []
    cnn_loss_min = []
    cnn_loss_mean = []
    cnn_loss_max = []
    cnn_loss_var = []
    don_loss_min = []
    don_loss_mean = []
    don_loss_max = []
    don_loss_var = []
    gir_time_min = []
    gir_time_mean = []
    gir_time_max = []
    gir_time_var = []
    cnn_time_min = []
    cnn_time_mean = []
    cnn_time_max = []
    cnn_time_var = []
    don_time_min = []
    don_time_mean = []
    don_time_max = []
    don_time_var = []
    em_time_min = []
    em_time_mean = []
    em_time_max = []
    em_time_var = []
    
    for N in np.arange(50,1000,50):
        dim = 10
        i = 20
        m=100
        p=15
        x0 = 0.1
        t0 = 0.
        X = 0.5
        T = i * 0.025
        num_time = 5 * i
        num_samples = 420
        batch_size = 5
        xs = torch.rand(num_samples,dim) * X + x0
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

        model = FKModule(m=m, p=p, X=X, t0=t0, T=T, batch_size=batch_size, dim=dim, num_time=num_time, N=N, n_batch_val=n_batch_val)
        trainer = pl.Trainer(max_epochs=1, gpus=1, check_val_every_n_epoch=1)
        trainer.fit(model, train_loader, val_loader)
        
        gir_loss_min.append(trainer.logged_metrics['gir_loss_min'].item())
        gir_loss_mean.append(trainer.logged_metrics['gir_loss_mean'].item())
        gir_loss_max.append(trainer.logged_metrics['gir_loss_max'].item())
        gir_loss_var.append(trainer.logged_metrics['gir_loss_var'].item())
        cnn_loss_min.append(trainer.logged_metrics['cnn_loss_min'].item())
        cnn_loss_mean.append(trainer.logged_metrics['cnn_loss_mean'].item())
        cnn_loss_max.append(trainer.logged_metrics['cnn_loss_max'].item())
        cnn_loss_var.append(trainer.logged_metrics['cnn_loss_var'].item())
        don_loss_min.append(trainer.logged_metrics['don_loss_min'].item())
        don_loss_mean.append(trainer.logged_metrics['don_loss_mean'].item())
        don_loss_max.append(trainer.logged_metrics['don_loss_max'].item())
        don_loss_var.append(trainer.logged_metrics['don_loss_var'].item())
        
        gir_time_min.append(trainer.logged_metrics['gir_time_min'].item())
        gir_time_mean.append(trainer.logged_metrics['gir_time_mean'].item())
        gir_time_max.append(trainer.logged_metrics['gir_time_max'].item())
        gir_time_var.append(trainer.logged_metrics['gir_time_var'].item())
        cnn_time_min.append(trainer.logged_metrics['cnn_time_min'].item())
        cnn_time_mean.append(trainer.logged_metrics['cnn_time_mean'].item())
        cnn_time_max.append(trainer.logged_metrics['cnn_time_max'].item())
        cnn_time_var.append(trainer.logged_metrics['cnn_time_var'].item())
        don_time_min.append(trainer.logged_metrics['don_time_min'].item())
        don_time_mean.append(trainer.logged_metrics['don_time_mean'].item())
        don_time_max.append(trainer.logged_metrics['don_time_max'].item())
        don_time_var.append(trainer.logged_metrics['don_time_var'].item())
        em_time_min.append(trainer.logged_metrics['em_time_min'].item())
        em_time_mean.append(trainer.logged_metrics['em_time_mean'].item())
        em_time_max.append(trainer.logged_metrics['em_time_max'].item())
        em_time_var.append(trainer.logged_metrics['em_time_var'].item())
        #print(trainer.logged_metrics['val_loss'])
        #print(trainer.logged_metrics['train_loss'])
        #ep = torch.arange(18)
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_cnn_loss_mean.npy', 'wb') as f:
            np.save(f, np.array(cnn_loss_mean))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_cnn_loss_min.npy', 'wb') as f:
            np.save(f, np.array(cnn_loss_min))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_cnn_loss_max.npy', 'wb') as f:
            np.save(f, np.array(cnn_loss_max))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_cnn_loss_var.npy', 'wb') as f:
            np.save(f, np.array(cnn_loss_var))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_gir_loss_mean.npy', 'wb') as f:
            np.save(f, np.array(gir_loss_mean))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_gir_loss_min.npy', 'wb') as f:
            np.save(f, np.array(gir_loss_min))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_gir_loss_max.npy', 'wb') as f:
            np.save(f, np.array(gir_loss_max))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_gir_loss_var.npy', 'wb') as f:
            np.save(f, np.array(gir_loss_var))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_don_loss_mean.npy', 'wb') as f:
            np.save(f, np.array(don_loss_mean))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_don_loss_min.npy', 'wb') as f:
            np.save(f, np.array(don_loss_min))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_don_loss_max.npy', 'wb') as f:
            np.save(f, np.array(don_loss_max))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_don_loss_var.npy', 'wb') as f:
            np.save(f, np.array(don_loss_var))
        
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_cnn_time_mean.npy', 'wb') as f:
            np.save(f, np.array(cnn_time_mean))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_cnn_time_min.npy', 'wb') as f:
            np.save(f, np.array(cnn_time_min))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_cnn_time_max.npy', 'wb') as f:
            np.save(f, np.array(cnn_time_max))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_cnn_time_var.npy', 'wb') as f:
            np.save(f, np.array(cnn_time_var))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_gir_time_mean.npy', 'wb') as f:
            np.save(f, np.array(gir_time_mean))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_gir_time_min.npy', 'wb') as f:
            np.save(f, np.array(gir_time_min))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_gir_time_max.npy', 'wb') as f:
            np.save(f, np.array(gir_time_max))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_gir_time_var.npy', 'wb') as f:
            np.save(f, np.array(gir_time_var))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_don_time_mean.npy', 'wb') as f:
            np.save(f, np.array(don_time_mean))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_don_time_min.npy', 'wb') as f:
            np.save(f, np.array(don_time_min))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_don_time_max.npy', 'wb') as f:
            np.save(f, np.array(don_time_max))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_don_time_var.npy', 'wb') as f:
            np.save(f, np.array(don_time_var))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_em_time_mean.npy', 'wb') as f:
            np.save(f, np.array(em_time_mean))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_em_time_min.npy', 'wb') as f:
            np.save(f, np.array(em_time_min))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_em_time_max.npy', 'wb') as f:
            np.save(f, np.array(em_time_max))
        with open('/scratch/xx84/girsanov/fbsde/ablation/result_N/N_bsb_em_time_var.npy', 'wb') as f:
            np.save(f, np.array(em_time_var))