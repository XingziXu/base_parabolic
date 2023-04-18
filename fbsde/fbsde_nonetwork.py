# simulation based on section 5 of https://www.sciencedirect.com/science/article/pii/S0005109817304740

import torch
from torch import nn
from torch.autograd import grad
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import pytorch_lightning as pl
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
from random import randint
import seaborn as sns 
import pandas as pd
import time

def b(t,x):
    return torch.sin(x)

def sigma(t,x):
    return torch.cos(torch.Tensor([t]))

def sigma_batch(t,x):
    return torch.cos(t)

def g(x):
    return torch.sin(x)

def h(t,x,y,z):
    return t+x+y+z

def divergence(y, x):
        div = 0.
        for i in range(y.shape[-1]):
            div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
        return div

def trace_grad(mu, Wt, N=10):

    # Hutchinson's trace trick

    dmu = 0
    for _ in range(N):
        #mu  = self.mu(Wt)
        v = torch.randn_like(mu) 
        dmu += (v * grad(mu, Wt, grad_outputs=v, create_graph=True)[0]).sum(-1).sum(-1) / N

    return dmu

if __name__ == '__main__':
    pl.seed_everything(1234)
    print(sys.executable)

    
    # EM calculation of FBSDE
    x_num = 60
    dim = 1
    N = 1000
    
    x0 = torch.linspace(0.,0.6, x_num).unsqueeze(1)#torch.rand(x_num, dim)
    t0 = torch.Tensor([0.8])
    T = 1.
    num_steps = 50
    t = torch.linspace(t0.item(), T, steps=num_steps)
    dt = t[1]-t[0]
    dB = np.sqrt(dt.item()) * torch.randn(N, dim, num_steps)
    dB = dB.unsqueeze(0).repeat(x_num,1,1,1)
    
    xi= x0.unsqueeze(1).repeat(1,N,1)
    xi = xi.unsqueeze(-1)
    xi.requires_grad = True
    for i in range(0,num_steps-1):
        x_current = xi[:,:,:,i] + b(t[i], xi[:,:,:,i]) * dt + sigma(t[i], xi[:,:,:,i]) * dB[:,:,:,i]
        xi = torch.cat((xi, x_current.unsqueeze(-1)),dim=-1)
    
    xT = xi[:,:,:,-1]
    yT = g(xT)
    yi = torch.zeros_like(xi)
    yi[:,:,:,-1] = yT
    vi = yT.mean(1)
    z_current = sigma(T,xT) * torch.autograd.grad(outputs=vi,inputs=xT,grad_outputs=torch.ones_like(vi))[0]
    for i in reversed(range(1,num_steps)):
        x_current = xi[:,:,:,i]
        t_current = t[i]
        yi[:,:,:,i-1] = yi[:,:,:,i] + h(t_current,x_current,yi[:,:,:,i],z_current) * dt
        vi = yi[:,:,:,i-1].mean(1)
        z_current = sigma(t_current,x_current) * torch.autograd.grad(outputs=vi,inputs=x_current,grad_outputs=torch.ones_like(vi))[0]
        
    vis = yi.mean(1)
    
    # change of PDE calculation of FBSDE
    t = torch.linspace(t0.item(), T, steps=num_steps)
    dt = t[1]-t[0]
    dB = np.sqrt(dt.item()) * torch.randn(N, dim, num_steps)
    dB = dB.unsqueeze(0).repeat(x_num,1,1,1)
    
    #xi= x0.unsqueeze(1).repeat(1,N,1)
    #xi = xi.unsqueeze(-1)
    #xi.requires_grad = True
    #for i in range(0,num_steps-1):
    #    x_current = xi[:,:,:,i] + sigma(t[i], xi[:,:,:,i]) * dB[:,:,:,i]
    #    xi = torch.cat((xi, x_current.unsqueeze(-1)),dim=-1)
        
    
    xi = torch.cumsum(dB * sigma_batch(t,x0).unsqueeze(0).unsqueeze(0).unsqueeze(0),dim=-1) + x0.unsqueeze(-1).unsqueeze(-1).repeat(1,N,1,num_steps)
    xi.requires_grad = True
    
    xT = xi[:,:,:,-1]
    yT = g(xT)
    yi = torch.zeros_like(xi)
    yi[:,:,:,-1] = yT
    vi = yT.mean(1)
    z_current = sigma(T,xT) * torch.autograd.grad(outputs=vi,inputs=xT,grad_outputs=torch.ones_like(vi))[0]
    for i in reversed(range(1,num_steps)):
        x_current = xi[:,:,:,i]
        t_current = t[i]
        yi[:,:,:,i-1] = yi[:,:,:,i] + (h(t_current,x_current,yi[:,:,:,i],z_current) + z_current * (1/sigma(t_current, x_current)) * b(t_current, x_current)) * dt
        vi = yi[:,:,:,i-1].mean(1)
        z_current = sigma(t_current,x_current) * torch.autograd.grad(outputs=vi,inputs=x_current,grad_outputs=torch.ones_like(vi))[0]
        
    vis_cov = yi.mean(1)
    
    # Girsanov calculation of FBSDE
    t = torch.linspace(t0.item(), T, steps=num_steps)
    dt = t[1]-t[0]
    dB = np.sqrt(dt.item()) * torch.randn(N, dim, num_steps)
    dB = dB.unsqueeze(0).repeat(x_num,1,1,1)
    
    #xi= x0.unsqueeze(1).repeat(1,N,1)
    #xi = xi.unsqueeze(-1)
    #xi.requires_grad = True
    #for i in range(0,num_steps-1):
    #    x_current = xi[:,:,:,i] + sigma(t[i], xi[:,:,:,i]) * dB[:,:,:,i]
    #    xi = torch.cat((xi, x_current.unsqueeze(-1)),dim=-1)
        
    sigmas = sigma_batch(t,x0)
    xi = torch.cumsum(dB * sigmas.unsqueeze(0).unsqueeze(0).unsqueeze(0),dim=-1) + x0.unsqueeze(-1).unsqueeze(-1).repeat(1,N,1,num_steps)
    xi.requires_grad = True
    
    sigmas = sigmas.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(x_num, N, dim, 1)
    mu = b(t, xi)/sigmas
    
    mart = torch.cumsum(mu * dB, dim=-1) - 0.5 * torch.cumsum(mu ** 2, dim=-1) * dt
    expmart = torch.exp(mart)
    
    xT = xi[:,:,:,-1]
    yT = g(xT)
    yi = torch.zeros_like(xi)
    yi[:,:,:,-1] = yT
    vi = yT.mean(1)
    z_current = sigma(T,xT) * torch.autograd.grad(outputs=vi,inputs=xT,grad_outputs=torch.ones_like(vi))[0]
    for i in reversed(range(1,num_steps)):
        x_current = xi[:,:,:,i]
        t_current = t[i]
        yi[:,:,:,i-1] = yi[:,:,:,i] + h(t_current,x_current,yi[:,:,:,i],z_current) * dt
        vi = yi[:,:,:,i-1].mean(1)
        z_current = sigma(t_current,x_current) * torch.autograd.grad(outputs=vi,inputs=x_current,grad_outputs=torch.ones_like(vi))[0]
        
    vis_gir = (yi * expmart).mean(1)
    
    plt.subplot(2,2,1)
    plt.imshow(vis.squeeze().detach().numpy())
    plt.colorbar()
    
    plt.subplot(2,2,2)
    plt.imshow(vis_gir.squeeze().detach().numpy())
    plt.colorbar()
    
    plt.subplot(2,2,3)
    plt.imshow(torch.abs(vis-vis_cov).squeeze().detach().numpy())
    plt.colorbar()
    
    plt.subplot(2,2,4)
    plt.imshow(torch.abs(vis-vis_gir).squeeze().detach().numpy())
    plt.colorbar()
    plt.show()