import numpy as np
import torch

def f(x):
    return (x**2).sum(-1)

T = 1
N = 1000
h = 0.01
d = 2

N_t = int(T / h)

Wt = (np.sqrt(h) * torch.randn(N, N_t, d, requires_grad=True)).cumsum(1)
WT = Wt[:,-1,:]

Y = torch.zeros(N, N_t)
YT = f(WT)
Y[:, -1] = YT

Zt = torch.autograd.grad(YT, WT, grad_outputs=torch.ones_like(YT), create_graph=True, retain_graph=True)[0]


for idx in range(1, N_t):
    Wti = Wt[:, idx-1, :]

    Y[:, idx] = Y[:, idx - 1] + (Zt ** 2).sum(-1)
    Yi = Y[:, idx]

    Zti = torch.autograd.grad(Yi, Wt[:, idx-1, :], grad_outputs=torch.ones_like(Yi), create_graph=True, retain_graph=True)[0]
