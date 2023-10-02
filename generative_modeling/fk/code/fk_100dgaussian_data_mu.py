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
import json

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
    def __init__(self, mean_rn):
        super().__init__()
        self.mean_rn = mean_rn
        #self.mu = MLP(input_dim=2, index_dim=0, hidden_dim=256, output_dim=2) # function representing the measure
        self.mu = SMLP(hyper_param['dim'], 200, 3, hyper_param['dim'], nn.Tanh())
        #self.sigma = MLP(input_dim=0, index_dim=1, hidden_dim=64, output_dim=1) # function representing the measure
        self.T = nn.Parameter(torch.Tensor([0.1])) # terminal time
        #self.T = 0.1 # terminal time
        self.num_steps = 40 # dt
        self.t = torch.linspace(0., self.T.data[0], self.num_steps-1).to(device) # time steps
        self.dt = self.t[1]-self.t[0] # size of time steps
        self.N = 75 # number of instances of Y_t used to approximate the expectation
        self.sdt = torch.sqrt(self.dt).to(device)
        self.batch_size = hyper_param['batch_size']
        self.dW = torch.randn(self.N, self.batch_size, self.num_steps-2, hyper_param['dim']).to(device)
        #self.p0 = MultivariateNormal(torch.zeros(2).to(device), torch.eye(2).to(device))
        self.num_steps_val = 50
        self.t_val = torch.linspace(0., self.T.data[0], self.num_steps_val-1).to(device) 
        self.dt_val = self.t_val[1]-self.t_val[0] 
        self.sdt_val = torch.sqrt(self.dt_val).to(device)
        #self.variance = nn.Parameter(torch.eye(2).to(device))
        self.weight_pos = 1e-2
        self.variance = torch.eye(hyper_param['dim']).to(device)
        #self.variance = SMLP(1, 16, 3, 2, nn.Tanh())
        
        # Define 2D Gaussian base distribution
        base = nf.distributions.base.DiagGaussian(hyper_param['dim'])

        # Define list of flows
        num_layers = 32
        flows = []
        for i in range(num_layers):
            # Neural network with two hidden layers having 64 units each
            # Last layer is initialized by zeros making training more stable
            #param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
            param_map = SMLP(int(hyper_param['dim']/2), 128, 3, hyper_param['dim'], nn.Tanh())
            # Add flow layer
            flows.append(nf.flows.AffineCouplingBlock(param_map))
            # Swap dimensions
            flows.append(nf.flows.Permute(2, mode='swap'))
            
        # Construct flow model
        model = nf.NormalizingFlow(base, flows) 
        model.load_state_dict(torch.load('/scratch/xx84/girsanov/generative_modeling/nf/trained_model/nf_prior_100.pt'))
        self.p0 = model
    
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
        ys = torch.zeros(self.num_steps, self.N, self.batch_size, y0.shape[1]).to(device)
        for i in range(self.num_steps-2):
            s_current = self.T-self.t[i]
            #sigma = torch.diag_embed(self.variance(s_current).unsqueeze(0).unsqueeze(0).repeat(self.N,self.batch_size,1))
            #sc = torch.Tensor([self.T - i * self.dt]).to(device)
            #s_current = (sc * torch.ones(y_current.shape[0],y_current.shape[1],1).to(device))
            #input_current = torch.cat((y_current, s_current),dim=2)
            #mu_current = self.mu(input_current).to(device)
            mu_current = self.mu(y_current).to(device)
            div = div - self.dt * trace_grad(mu_current, y_current).squeeze()
            #div = div + self.dt * divergence(mu_current, y_current).squeeze()
            #sigma_current = self.sigma(sc) * torch.ones(y_current.shape[0], y_current.shape[1], y_current.shape[2], y_current.shape[2]).to(device)
            dW_current = dW[:,:,i,:]#.unsqueeze(-1).repeat(y_current.shape[0],1,1,1)
            #y_current = y_current - mu_current * self.dt + torch.matmul(self.variance,dW_current.unsqueeze(-1)).squeeze()#.squeeze()#(torch.matmul(sigma_current, dW_current)).squeeze()
            y_current = y_current + mu_current * self.dt + 0.1 * dW_current.squeeze()#.squeeze()#(torch.matmul(sigma_current, dW_current)).squeeze()
            ys[i, :, :, :] = y_current
        ys = torch.cumsum((ys ** 2).sum(-1) * self.dt, dim=0)[-1,:,:]
        yT = y_current
        #logp0yT = self.p0.log_prob(yT)
        #pxt = torch.mean(torch.exp(logp0yT) * div)
        
        return - self.elbo(yT,div) #+ torch.mean(ys) * 0.5# - self.weight_pos * pos_def(self.variance, N=10)

    def elbo(self,yT,div):
        logp0yT = self.p0.log_prob(yT.reshape(yT.shape[0]*yT.shape[1],yT.shape[2]))
        elbo = torch.mean(logp0yT) + torch.mean(div)
        return elbo
    
    def exact_prob(self,yT,div):
        logp0yT = self.p0.log_prob(yT.reshape(yT.shape[0]*yT.shape[1],yT.shape[2]))
        pxt = torch.mean(torch.exp(logp0yT) * torch.exp(div))
        return pxt

    def sample(self, num_sample):
        with torch.no_grad():
            x = self.p0.sample(num_sample)[0]
            x_current = x
            self.t_val = torch.linspace(0., self.T.data[0], self.num_steps_val-1).to(device) # time steps
            self.dt_val = self.t_val[1]-self.t_val[0]
            self.sdt_val = torch.sqrt(self.dt_val).to(device)
            dW = self.sdt_val * torch.randn(num_sample, self.num_steps_val-2, hyper_param['dim']).to(device)
            for i in range(self.num_steps_val-2):
                s_current = self.T-self.t_val[i]
                #sigma = torch.diag_embed(self.variance.unsqueeze(0).repeat(num_sample,1))
                #sc = torch.Tensor([i * self.dt]).to(device)
                #s_current = (sc * torch.ones(y_current.shape[0],1).to(device))
                #input_current = torch.cat((y_current, s_current),dim=1)
                mu_current = self.mu(x_current).to(device)
                #sigma_current = self.sigma(sc) * torch.ones(y_current.shape[0], y_current.shape[1], y_current.shape[1]).to(device)
                dW_current = dW[:,i,:]#.unsqueeze(-1)
                #x_current = x_current + mu_current * self.dt_val + torch.matmul(self.variance,dW_current.unsqueeze(-1)).squeeze()#.squeeze()#(torch.matmul(sigma_current, dW_current)).squeeze()
                x_current = x_current + mu_current * self.dt_val + 0.1 * dW_current.squeeze()#.squeeze()#(torch.matmul(sigma_current, dW_current)).squeeze()
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
        plt.scatter(batch[:,0], batch[:,1], alpha=0.1, label='Real')
        plt.scatter(xT[:,0], xT[:,1], alpha=0.1, label=r'$X_T$')
        plt.scatter(x0[:,0], x0[:,1], alpha=0.1, label=r'$X_0$')
        plt.legend()
        plt.savefig('/scratch/xx84/girsanov/generative_modeling/fk/figure/fokker_planck_elbo.png')
        plt.clf()
        torch.save(self.mu.state_dict(), '/scratch/xx84/girsanov/generative_modeling/fk/trained_model/dim_100/mu_'+str(self.mean_rn)+'.pt')
        """
        xs = torch.linspace(-2, 3, steps=100)
        ys = torch.linspace(-1, 2, steps=100)
        xy = torch.Tensor(torch.Tensor([[1.,1.]]))
        for x in xs:
            for y in ys:
                xy = torch.cat((xy,torch.cat((torch.Tensor([x]),torch.Tensor([y]))).unsqueeze(dim=0)),dim=0)
        xy = xy[1:,:]
        y0 = xy
        y_current = y0.unsqueeze(1).repeat(1,self.N,1).to(device)
        y_current.requires_grad = True
        div = torch.zeros(y0.shape[0],self.N).to(device)
        for i in range(self.num_steps):
            sc = torch.Tensor([self.T - i * self.dt]).to(device)
            s_current = (sc * torch.ones(y_current.shape[0],y_current.shape[1],1).to(device))
            input_current = torch.cat((y_current, s_current),dim=2)
            mu_current = self.mu(input_current).to(device)
            div = div - self.dt * divergence(mu_current, y_current).squeeze()
            sigma_current = self.sigma(sc) * torch.ones(y_current.shape[0], y_current.shape[1], y_current.shape[2], y_current.shape[2]).to(device)
            dW_current = self.dW[:,i,:].unsqueeze(0).unsqueeze(-1).repeat(y_current.shape[0],1,1,1)
            y_next = y_current - mu_current * self.dt + (torch.matmul(sigma_current, dW_current)).squeeze()
            y_current = y_next
        yT = y_current
        prob = self.exact_prob(yT,div)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xy[:,0].detach().cpu().numpy(), xy[:,1].detach().cpu().numpy(), prob.detach().cpu().numpy(), c = 'b', marker='o')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.savefig('something.png')
        """
        return {'val_loss': val_loss}
        
        
    def configure_optimizers(self):
        #optimizer = torch.optim.AdamW([{'params': self.mu.parameters()},{'params': self.variance.parameters()}], lr=8e-4)
        optimizer = torch.optim.AdamW([{'params': self.mu.parameters()}], lr=hyper_param['learning_rate'])
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
    
    
    base_dir = '/scratch/xx84/girsanov/'# Define 2D Gaussian base distribution
    json_data = open(base_dir+"generative_modeling/fk/code/gaussian_100.json", "r", encoding="utf-8")
    hyper_param = json.load(json_data)
    json_data.close()

    base = nf.distributions.base.DiagGaussian(hyper_param['dim'])

    # Define list of flows
    num_layers = 32
    flows = []
    for i in range(num_layers):
        # Neural network with two hidden layers having 64 units each
        # Last layer is initialized by zeros making training more stable
        #param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
        param_map = SMLP(int(hyper_param['dim']/2), 128, 3, hyper_param['dim'], nn.Tanh())
        # Add flow layer
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        # Swap dimensions
        flows.append(nf.flows.Permute(2, mode='swap'))
        
    # Construct flow model
    model = nf.NormalizingFlow(base, flows)
    model.load_state_dict(torch.load('/scratch/xx84/girsanov/generative_modeling/nf/trained_model/nf_prior_100.pt'))
    p0 = model
    d0 = p0.sample(500)[0].detach().numpy()
    Loss =  SamplesLoss("sinkhorn", blur=0.05,)
    loss = []

    num_samples = 10000

    p0 = MultivariateNormal(torch.ones(hyper_param['dim']) * 0.0, torch.eye(hyper_param['dim']))
    p1 = MultivariateNormal(torch.ones(hyper_param['dim']) * 1.0, torch.eye(hyper_param['dim']))
    p2 = MultivariateNormal(torch.ones(hyper_param['dim']) * 2.0, torch.eye(hyper_param['dim']))
    p3 = MultivariateNormal(torch.ones(hyper_param['dim']) * 3.0, torch.eye(hyper_param['dim']))
    p4 = MultivariateNormal(torch.ones(hyper_param['dim']) * 4.0, torch.eye(hyper_param['dim']))
    p5 = MultivariateNormal(torch.ones(hyper_param['dim']) * 5.0, torch.eye(hyper_param['dim']))
    p6 = MultivariateNormal(torch.ones(hyper_param['dim']) * 6.0, torch.eye(hyper_param['dim']))
    p7 = MultivariateNormal(torch.ones(hyper_param['dim']) * 7.0, torch.eye(hyper_param['dim']))
    p8 = MultivariateNormal(torch.ones(hyper_param['dim']) * 8.0, torch.eye(hyper_param['dim']))
    p9 = MultivariateNormal(torch.ones(hyper_param['dim']) * 9.0, torch.eye(hyper_param['dim']))
    x0 = p0.sample([int(num_samples/10)]).data.numpy()
    x1 = p1.sample([int(num_samples/10)]).data.numpy()
    x2 = p2.sample([int(num_samples/10)]).data.numpy()
    x3 = p3.sample([int(num_samples/10)]).data.numpy()
    x4 = p4.sample([int(num_samples/10)]).data.numpy()
    x5 = p5.sample([int(num_samples/10)]).data.numpy()
    x6 = p6.sample([int(num_samples/10)]).data.numpy()
    x7 = p7.sample([int(num_samples/10)]).data.numpy()
    x8 = p8.sample([int(num_samples/10)]).data.numpy()
    x9 = p9.sample([int(num_samples/10)]).data.numpy()
    trainset = np.concatenate((x0,x1,x2,x3,x4,x5,x6,x7,x8,x9),axis=0)

    x0 = p0.sample([int(num_samples/50)]).data.numpy()
    x1 = p1.sample([int(num_samples/50)]).data.numpy()
    x2 = p2.sample([int(num_samples/50)]).data.numpy()
    x3 = p3.sample([int(num_samples/50)]).data.numpy()
    x4 = p4.sample([int(num_samples/50)]).data.numpy()
    x5 = p5.sample([int(num_samples/50)]).data.numpy()
    x6 = p6.sample([int(num_samples/50)]).data.numpy()
    x7 = p7.sample([int(num_samples/50)]).data.numpy()
    x8 = p8.sample([int(num_samples/50)]).data.numpy()
    x9 = p9.sample([int(num_samples/50)]).data.numpy()
    
    testset = np.concatenate((x0,x1,x2,x3,x4,x5,x6,x7,x8,x9),axis=0)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size = hyper_param['batch_size'], shuffle=True, num_workers = 1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size = hyper_param['batch_size'], shuffle=True)
    #train_loader = torch.utils.data.DataLoader(trainset, batch_size = 256, shuffle=True, num_workers = 1)
    #test_loader = torch.utils.data.DataLoader(testset, batch_size = 1000, shuffle=False)
    model = FKModule(mean_rn = 20)
    trainer = pl.Trainer(max_epochs=20,gpus=1)
    trainer.fit(model, train_loader, test_loader)
    #val_bpd.append(trainer.logged_metrics['val_loss'])
    loss.append(Loss(torch.tensor(d0).type(torch.FloatTensor),torch.tensor(trainset).type(torch.FloatTensor)).item())
