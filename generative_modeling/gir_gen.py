import torch
from torch import nn
from torch.autograd import grad
from torch.nn import functional as F

import numpy as np

import pytorch_lightning as pl

from torchvision.utils import make_grid

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

D_IDX = -1
T_IDX = -2
N_IDX =  0

class FKModule(pl.LightningModule):
    def __init__(self, d, mu, g, T = 1, dt = 0.05, N = 100, lr_mu = 1e-2):
        super().__init__()

        # dimension
        self.d = d

        # N expectation
        self.N = N

        # boundary and drift functions
        self.mu = mu
        self.g  = g

        # integration time
        t   = torch.linspace(0, T, int(T/dt))
        self.register_buffer('t', t)

        # N of expectation x 1 x T x d
        dWt = torch.randn(N, 1, self.t.shape[0], d) * np.sqrt(dt)
        self.register_buffer('dWt', dWt)

        self.lr_mu = lr_mu

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.reshape(-1, self.d)
        return -self.log_p_E(x)

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        samples, xinit = self.sample(x)

        samples = samples.detach().cpu()
        xinit = xinit.detach().cpu()
        x0 = x.detach().cpu()

        if x.shape[-1] == 2:
            # plot the points
            plt.scatter(x0[:,0], x0[:,1], alpha=0.1, label='Real')
            plt.scatter(samples[:,0], samples[:,1], alpha=0.1, label=r'$X_T$')
            plt.scatter(xinit[:,0], xinit[:,1], alpha=0.1, label=r'$X_0$')
            plt.legend()
            plt.savefig('samples.pdf')
            plt.close('all')

            # plot the vector fields 
            x,y = torch.meshgrid(torch.linspace(-3,3,20).type_as(x), torch.linspace(-3,3,20).type_as(x))
            X_ = torch.stack((x,y),-1).reshape(-1,2)

            plt.quiver(x.cpu(), y.cpu(), -self.mu(X_)[:,0].cpu(), -self.mu(X_)[:,1].cpu())
            plt.savefig('negvec.pdf')
            plt.close('all')

            plt.quiver(x.cpu(), y.cpu(), self.mu(X_)[:,0].cpu(), self.mu(X_)[:,1].cpu())
            plt.savefig('vec.pdf')
            plt.close('all')
        else:
            samples = samples.reshape(-1,int(np.sqrt(self.d)), int(np.sqrt(self.d)))
            grid = make_grid(samples.unsqueeze(1))

            plt.imshow(grid[0], cmap='gray')
            plt.savefig('samples.png')
            plt.close('all')

    def log_p_E(self, x):
        # x shape is batch size x d
        dt = self.t[1] - self.t[0]

        #dWt = (torch.randn(self.N, 1, self.t.shape[0], self.d).type_as(x) * torch.sqrt(dt)) # TODO: check difference
        dWt = self.dWt

        Wt  = dWt.cumsum(-2) + x.unsqueeze(1) # compute brownian path up to time T
        Wt[:,:,0,:] = x

        Wt.requires_grad = True

        mu  = self.mu(Wt)

        tr_dmu = self.trace_grad(mu, Wt)  # trace of jacobian
        v = tr_dmu

        a = (mu * dWt).sum(-1).sum(-1)                # first term in girsanov
        b = - 1/2 * dt * ( (mu ** 2).sum(-1)).sum(-1) # second term in girsanov

        g = (self.g(Wt[:,:,-1,:])).log()              # boundary condition 

        #plt.scatter(Wt[:,:,-1,0].reshape(-1,1).cpu().detach(), Wt[:,:,-1,1].reshape(-1,1).cpu().detach())
        #plt.scatter(Wt[:,:,0,0].reshape(-1,1).cpu().detach(), Wt[:,:,0,1].reshape(-1,1).cpu().detach())
        #plt.savefig('finalt.pdf')
        #plt.close('all')
        #exit()

        return ( g + a + b + v ).mean()

    def trace_grad(self, mu, Wt, N=10):

        # Hutchinson's trace trick

        dmu = 0
        for _ in range(N):
            mu  = self.mu(Wt)
            v = torch.randn_like(mu) 
            dmu += (v * grad(mu, Wt, grad_outputs=v, create_graph=True)[0]).sum(-1).sum(-1) / N

        return dmu

    def sample(self, x0, N=500, dt=0.0001):

        # basic Euler-Maruyama routine

        T = self.t[-1]
        Nt = int(T/dt)

        x0_ = torch.randn(N, self.d).type_as(self.dWt)  #*0.01
        x0 = x0_.clone()

        t_fine = torch.linspace(0,T,Nt)

        for idx, _ in enumerate(t_fine):

            x1 = x0 - self.mu(x0) * dt +  np.sqrt(dt) * torch.randn(N,self.d).to(self.dWt.device) #self.dWt[:N,0,idx,:]
            x0 = x1

        return x1, x0_

    def configure_optimizers(self):

        opt_params = [{'params': self.mu.parameters(), 'lr': self.lr_mu}]

        optimizer = torch.optim.Adam(opt_params)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        # TODO: see if these help

        return {'optimizer' : optimizer}#, 'scheduler': scheduler, 'monitor' : 'train_loss'}