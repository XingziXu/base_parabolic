# this code is based on equation (11) in the paper https://arxiv.org/pdf/2106.02808.pdf
# we define a forward sde dX=mu(x,t)dt+sigma(t)dB, which subsequently defines a sequence
# of distributions. We optimize mu and sigma so that p(d_i,T) is maximized, where p(d_i,T)
# is the probability of data points at terminal time T.
import torch
from torch import nn
from torch.autograd import grad
from torch.nn import functional as F
import os
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
from geomloss import SamplesLoss
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
import time
from torchvision.utils import make_grid
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.datasets import make_moons
import sys
import math
import normflows as nf
from scipy.stats.stats import pearsonr

class TwoGaussian:
    def sample(self, n):
        a = torch.randn(int(n/2),2)*0.1 - 0.5
        b = torch.randn(int(n/2),2)*0.1 + 0.5
        return torch.cat((a,b))

class SkewGaussian:
    def sample(self, n):
        p = MultivariateNormal(torch.zeros(2).to(device), torch.Tensor([[10,0],[0.,0.01]]).to(device))
        return p.sample((n,))

class Circ:
    def sample(self, n, noise=0.05):
        if noise is None:
            noise = 0.05
        theta = torch.rand(n) * 2 * math.pi
        r = torch.Tensor([1]) + torch.randn(n) * noise
        x = (r * torch.cos(theta)).unsqueeze(1)
        y = (r * torch.sin(theta)).unsqueeze(1)
        return torch.cat((x,y),dim=1)

class Moons:
    """
    Swiss roll distribution sampler.
    noise control the amount of noise injected to make a thicker swiss roll
    """
    def sample(self, n, noise=0.05):
        if noise is None:
            noise = 0.05
        return torch.from_numpy(
            make_moons(n_samples=n, noise=noise)[0]*1.5)
        
        
class SwissRoll:
    """
    Swiss roll distribution sampler.
    noise control the amount of noise injected to make a thicker swiss roll
    """
    def sample(self, n, noise=0.5):
        if noise is None:
            noise = 0.5
        return torch.from_numpy(
            make_swiss_roll(n_samples=n, noise=noise)[0][:, [0, 2]].astype('float32') / 5.)
        
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x): 
        return torch.sigmoid(x)*x

class MLP(nn.Module):
    def __init__(self,
                 input_dim=2,
                 index_dim=1,
                 hidden_dim=128,
                 output_dim=2,
                 act=nn.CELU(),
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.index_dim = index_dim
        self.hidden_dim = hidden_dim
        self.act = act

        self.main = nn.Sequential(
            nn.Linear(input_dim+index_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, input):
        # init
        #sz = input.size()
        #input = input.view(-1, self.input_dim)
        #t = t.view(-1, self.index_dim).float()

        # forward
        #h = torch.cat([input, t], dim=1) # concat
        output = self.main(input) # forward
        return output
    
class SMLP_nf(nn.Module):
    def __init__(self, input_size, hidden_size, layers, out_size, act=nn.SELU()):
        super(SMLP_nf, self).__init__()

        self.act = act

        self.fc1 = nn.Linear(input_size,hidden_size)
        mid_list = []
        for i in range(layers):
           mid_list += [nn.Linear(hidden_size,hidden_size), act]

        self.mid = nn.Sequential(*mid_list)
        self.out = nn.Linear(hidden_size, out_size, bias=False)

    def forward(self,x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.mid(out)
        out = self.out(out)
        return out    

class SMLP(nn.Module):
    def __init__(self, input_size, hidden_size, layers, out_size, act=nn.SELU()):
        super(SMLP, self).__init__()

        self.act = act

        self.fc1 = nn.Linear(input_size,hidden_size)
        mid_list = []
        for i in range(layers):
           mid_list += [nn.Linear(hidden_size,hidden_size), act]

        self.mid = nn.Sequential(*mid_list)
        self.out = nn.Linear(hidden_size, out_size, bias=False)

    def forward(self,x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.mid(out)
        out = self.out(out)
        return out    
    
def gradient(y, x, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True)[0]
    return grad


def jacobian(y, x):
    """Compute dy/dx = dy/dx @ grad_outputs; 
    for grad_outputs in [1, 0, ..., 0], [0, 1, 0, ..., 0], ...., [0, ..., 0, 1]"""
    jac = torch.zeros(y.shape[0], x.shape[0])
    grad_outputs = torch.zeros_like(y)
    for i in range(y.shape[0]):
        grad_outputs[i] = 1
        jac[i] = gradient(y, x, grad_outputs = grad_outputs)
        grad_outputs[i] = 0
    return jac

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)

def sample_rademacher(shape):
    return (torch.rand(*shape).ge(0.5)).float() * 2 - 1

def trace_grad(mu, Wt, N=2):

    # Hutchinson's trace trick

    dmu = 0
    for _ in range(N):
        #mu  = self.mu(Wt)
        v = torch.randn_like(mu) 
        #v = sample_rademacher(mu.shape).to(device)
        #v.requires_grad = True
        dmu += (v * grad(mu, Wt, grad_outputs=v, create_graph=True)[0]).sum(-1) / N

    return dmu

def pos_def(mat, N=10):
    v = torch.Arandn(N, mat.shape[0]).to(device)
    return torch.min(torch.matmul(torch.matmul(v.unsqueeze(1),mat),v.unsqueeze(-1)).squeeze())

class FKModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.dim = 100
        #self.mu = MLP(input_dim=2, index_dim=0, hidden_dim=256, output_dim=2) # function representing the measure
        self.mu = SMLP(self.dim, 200, 3, self.dim, nn.Tanh())
        #self.sigma = MLP(input_dim=0, index_dim=1, hidden_dim=64, output_dim=1) # function representing the measure
        self.T = nn.Parameter(torch.Tensor([0.1])) # terminal time
        #self.T = 0.1 # terminal time
        self.num_steps = 40 # dt
        self.t = torch.linspace(0., self.T.data[0], self.num_steps-1).to(device) # time steps
        self.dt = self.t[1]-self.t[0] # size of time steps
        self.N = 75 # number of instances of Y_t used to approximate the expectation
        self.sdt = torch.sqrt(self.dt).to(device)
        self.batch_size = 200
        self.dW = torch.randn(self.N, self.batch_size, self.num_steps-2, self.dim).to(device)
        #self.p0 = MultivariateNormal(torch.zeros(2).to(device), torch.eye(2).to(device))
        self.num_steps_val = 50
        self.t_val = torch.linspace(0., self.T.data[0], self.num_steps_val-1).to(device) 
        self.dt_val = self.t_val[1]-self.t_val[0] 
        self.sdt_val = torch.sqrt(self.dt_val).to(device)
        #self.variance = nn.Parameter(torch.eye(2).to(device))
        self.weight_pos = 1e-2
        #self.variance = torch.eye(2).to(device)
        self.variance = SMLP(1, 16, 3, self.dim, nn.Tanh())
        
        self.p0 = MultivariateNormal(torch.zeros(self.dim).to(device), torch.eye(self.dim).to(device))
    
    def loss(self, batch):
        # we calculate p(d_i,T) using equation (11). 
        # we need to calculate Y_T, and div(mu(Y_s),T-s) for s from t0=0 to t1=T
        # alternatively, we can maybe apply Jensen's inequality and use equation (13), the ELBO
        self.t = torch.linspace(0., self.T.data[0], self.num_steps-1).to(device) # time steps
        self.dt = self.t[1]-self.t[0]
        self.sdt = torch.sqrt(self.dt).to(device)
        dW = self.dW * self.sdt
        y0 = batch
        y_current = y0.unsqueeze(0).repeat(self.N,1,1).to(device)
        y_current.requires_grad = True
        div = torch.zeros(self.N,y0.shape[0]).to(device)
        for i in range(self.num_steps-2):
            s_current = self.T-self.t[i]
            sigma = torch.diag_embed(self.variance(s_current).unsqueeze(0).unsqueeze(0).repeat(self.N,self.batch_size,1))
            #sc = torch.Tensor([self.T - i * self.dt]).to(device)
            #s_current = (sc * torch.ones(y_current.shape[0],y_current.shape[1],1).to(device))
            #input_current = torch.cat((y_current, s_current),dim=2)
            #mu_current = self.mu(input_current).to(device)
            mu_current = self.mu(y_current).to(device)
            #div = div - self.dt * trace_grad(mu_current, y_current).squeeze()
            div = div + self.dt * trace_grad(mu_current, y_current).squeeze()
            #sigma_current = self.sigma(sc) * torch.ones(y_current.shape[0], y_current.shape[1], y_current.shape[2], y_current.shape[2]).to(device)
            dW_current = dW[:,:,i,:]#.unsqueeze(-1).repeat(y_current.shape[0],1,1,1)
            #y_current = y_current - mu_current * self.dt + torch.matmul(self.variance,dW_current.unsqueeze(-1)).squeeze()#.squeeze()#(torch.matmul(sigma_current, dW_current)).squeeze()
            y_current = y_current + mu_current * self.dt + torch.matmul(sigma,dW_current.unsqueeze(-1)).squeeze()#.squeeze()#(torch.matmul(sigma_current, dW_current)).squeeze()
        yT = y_current
        #logp0yT = self.p0.log_prob(yT)
        #pxt = torch.mean(torch.exp(logp0yT) * div)
        
        return -self.elbo(yT,div)# - self.weight_pos * pos_def(self.variance, N=10)

    def elbo(self,yT,div):
        logp0yT = self.p0.log_prob(yT)
        elbo = torch.mean(logp0yT) + torch.mean(div)
        return elbo
    
    def exact_prob(self,yT,div):
        logp0yT = self.p0.log_prob(yT)
        pxt = torch.mean(torch.exp(logp0yT) * torch.exp(div))
        return pxt

    def sample(self, num_sample):
        with torch.no_grad():
            x = self.p0.sample((num_sample,))
            x_current = x
            self.t_val = torch.linspace(0., self.T.data[0], self.num_steps_val-1).to(device) # time steps
            self.dt_val = self.t_val[1]-self.t_val[0]
            self.sdt_val = torch.sqrt(self.dt_val).to(device)
            dW = self.sdt_val * torch.randn(num_sample, self.num_steps_val-2, self.dim).to(device)
            for i in range(self.num_steps_val-2):
                s_current = self.T-self.t_val[i]
                sigma = torch.diag_embed(self.variance(s_current).unsqueeze(0).repeat(num_sample,1))
                #sc = torch.Tensor([i * self.dt]).to(device)
                #s_current = (sc * torch.ones(y_current.shape[0],1).to(device))
                #input_current = torch.cat((y_current, s_current),dim=1)
                mu_current = self.mu(x_current).to(device)
                #sigma_current = self.sigma(sc) * torch.ones(y_current.shape[0], y_current.shape[1], y_current.shape[1]).to(device)
                dW_current = dW[:,i,:]#.unsqueeze(-1)
                #x_current = x_current + mu_current * self.dt_val + torch.matmul(self.variance,dW_current.unsqueeze(-1)).squeeze()#.squeeze()#(torch.matmul(sigma_current, dW_current)).squeeze()
                x_current = x_current - mu_current * self.dt_val + torch.matmul(sigma,dW_current.unsqueeze(-1)).squeeze()#.squeeze()#(torch.matmul(sigma_current, dW_current)).squeeze()
        return x, x_current


    def training_step(self, batch, batch_idx):
        #torch.autograd.set_detect_anomaly(True)
        #optimizer_mu, optimizer_g = self.optimizers
        
        batch = batch.to(device)
        batch = batch.type(torch.FloatTensor)
        #x = x.reshape(-1, x.ndim)
        start = time.time()
        loss_sde = self.loss(batch)
        end = time.time()
        self.log('loss_time', end-start)
        self.log('train_loss', loss_sde)
        return {'loss': loss_sde}
    
    def validation_step(self, batch, batch_idx):
        
        super().validation_step(self, batch, batch_idx)
        torch.set_grad_enabled(True)
        x0,xT = self.sample(num_sample = 1000)
        x0 = x0.cpu().numpy()
        xT = xT.cpu().numpy()
        batch = batch.to(device)
        batch = batch.type(torch.FloatTensor)
        val_loss = self.loss(batch)
        self.log('val_loss', val_loss)
        batch = batch.cpu().numpy()
        return {'val_loss': val_loss}
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([{'params': self.mu.parameters()},{'params': self.variance.parameters()}], lr=8e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        #optimizer_g = torch.optim.AdamW([self.variance], lr=1e-5)
        #optimizers = [optimizer_mu, optimizer_g]
        #schedulers = [{"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_mu), "monitor": "train_loss"}, {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_g), "monitor": "train_loss"}]
        return {'optimizer' : optimizer, 'scheduler': scheduler, 'monitor' : 'train_loss'}
        #return optimizers, schedulers

if __name__ == '__main__':
    pl.seed_everything(1240)
    print(sys.executable)
    device = torch.device("cuda:0")
    
    dim=100
    
    # train model with cifar datasets of different labels
    val_bpd = []
    p0 = MultivariateNormal(torch.zeros(100).to(device), torch.eye(100).to(device))
    d0 = p0.sample((500,)).cpu().detach().numpy()
    Loss =  SamplesLoss("sinkhorn", blur=0.05,)
    loss = []

    for i in range(10):
        p_i = MultivariateNormal(torch.zeros(dim) + 1. * i * torch.ones(dim), torch.eye(dim))
        trainset = p_i.sample([2000])
        testset = p_i.sample([200])
        train_loader = torch.utils.data.DataLoader(trainset, batch_size = 200, shuffle=True, num_workers = 1)
        test_loader = torch.utils.data.DataLoader(testset, batch_size = 200, shuffle=True)
        #train_loader = torch.utils.data.DataLoader(trainset, batch_size = 256, shuffle=True, num_workers = 1)
        #test_loader = torch.utils.data.DataLoader(testset, batch_size = 1000, shuffle=False)
        model = FKModule()
        trainer = pl.Trainer(max_epochs=3,gpus=1)
        trainer.fit(model, train_loader, test_loader)
        val_bpd.append(trainer.logged_metrics['val_loss'])
        loss.append(Loss(torch.tensor(d0).type(torch.FloatTensor),torch.tensor(trainset).type(torch.FloatTensor)).item())

    with open('/scratch/xx84/girsanov/generative_modeling/100dgaussian_bpd_0.npy', 'wb') as f:
        np.save(f, val_bpd)
    with open('/scratch/xx84/girsanov/generative_modeling/100dgaussian_loss_0.npy', 'wb') as f:
        np.save(f, loss)
    fig = plt.figure()
    ax0 = fig.add_subplot(111)
    ax0.scatter(loss, val_bpd)
    ax0.set_ylabel('ELBO Loss')
    #ax1 = fig.add_subplot(212)
    #ax1.plot(loss[2:], loss_time[2:])
    #ax1.scatter(loss[0:2], loss_time[0:2])
    #ax1.set_ylabel('integration time')
    ax0.set_xlabel('Wasserstein distance')
    plt.savefig('bpd_toy_fokker_planck_100d.png')
    print(pearsonr(loss, val_bpd))