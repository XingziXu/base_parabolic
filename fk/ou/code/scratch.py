import numpy as np
np.random.seed(42)

from scipy import interpolate

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm

import pickle

from timeit import default_timer as timer

import torch
from torchvision.utils import make_grid

from joblib import Parallel, delayed

font = {'size'   : 26}
matplotlib.rc('font', **font)


# The goal is to: 
# a) solve an equation using Feynman-Kac
# b) compute a change of measure to compute the solution to a new PDE

def rho_parabolic(mu2, mu, s, W, dt, dB = None):
    '''
    W is shape N x x-grid x time
    '''
    f = (mu2(W, 0) - mu(W, 0) / s(W, 0))
    f[:,:,0] = 0
    if dB is None:
        dB = np.sqrt(dt) * np.random.randn(*W.shape)
    norm = ( f ** 2 * dt).cumsum(-1) 
    a = -0.5 * norm
    b = (f * dB).cumsum(-1)
    a[:,:,0] = 0
    b[:,:,0] = 0

    interior = a + b

    #interior[interior > 6] = 0

    return np.exp(interior)

def ou_exact(x,t, A=-1, K=1000):

    x = np.expand_dims(x,0).repeat(K,0)
    t = np.expand_dims(t,0).repeat(K,0)

    dt = t[0,0,1]  - t[0,0,0]

    s = (dt * np.ones_like(t)).cumsum(-1)

    dB = np.sqrt(dt) * np.random.randn(*x.shape)
    dB[:,:,0] = 0 

    z = np.exp(t*A)*x + (np.exp((t - s) * A) * dB).cumsum(-1)

    return z, dB

def euler_parabolic(t, x0, m, s, dW=None, K=1000):
    '''
    Solves the parabolic probelm using Euler-Maruyama
    t : time points to solve
    x0: starting x positions
    m : drift
    s : diffusion
    dW : brownian motion
    K : number of expectations to take over
    '''
    N = t.shape[0]
    dt = t[1]-t[0]
    sdt = np.sqrt(dt)

    xt = np.zeros((K, x0.shape[0], N))

    xt[:, :, 0] = x0.copy()

    if dW is None:
        dW = np.sqrt(dt) * np.random.randn(*xt.shape)

    for idx in range(xt.shape[2]-1):
      
        ti = t[idx]
        xi = xt[:,:,idx]
        xt[:,:,idx+1] = xi + m(xi,ti) * dt + s(xi,ti) * dW[:,:,idx]

    return xt, dW

def parabolic():

    def g(x):
        return np.abs(np.sin(6*x))

    def f(x):
        return np.exp(x)

    def f_inv(x):
        return np.log(x)

    def mu(x,t):
        return -x/2

    def mu2(x,t):
        return 3 * np.ones_like(x) + x

    def mu3(x,t):
        return np.zeros_like(x)

    def s(x,t):
        return x

    T = 0.05
    n_K = 10000

    t = np.linspace(0, T, 250)
    dt =  t[1] - t[0]
    x = np.linspace(0.1, 1, 250)
    xexact = f_inv(x)

    import timeit

    dB = np.sqrt(dt) * np.random.randn(n_K, x.shape[0], t.shape[0])
    dB[:,:,0] = 0 

    start_time = timeit.default_timer()
    B = dB.copy()
    B[:,:,0] = xexact.copy()
    B = B.cumsum(-1)
    Bsig = f(B)
    dBsig = np.zeros_like(Bsig)
    dBsig[:,:,1:] = Bsig[:,:,1:] - Bsig[:,:,:-1]
    exs = g(Bsig).mean(0)
    print('Exact BM sim time: {}'.format(timeit.default_timer() - start_time))

    biased = ( g(Bsig) * (rho_parabolic(mu2, mu, s, B, dt, dB))).mean(0)
    #biased = ( g(Bsig) * (rho_parabolic(mu2, mu, s, B, dt, dB))).mean(0)
    #biased = ( g(Bsig) ).mean(0)
    print(biased.max())
    print(biased.min())
    print('Biased sim total time: {}'.format(timeit.default_timer() - start_time))

    cmap = 'YlGnBu'
    mapname = 'YlGnBu'

    start_time = timeit.default_timer()
    zt, dW = (euler_parabolic(t, x, mu2, s, dB, K=n_K))
    euler = g(zt)
    print(euler.mean(0).max())
    print(euler.mean(0).min())
    print('Euler time: {}'.format(timeit.default_timer() - start_time))

    plt.imshow(euler.mean(0), cmap=mapname)
    plt.savefig('em.png')
    plt.close('all')

    euler_com = ( g(zt) * rho_parabolic(mu2, mu3, s, zt, dt, dW) ).mean(0)
    plt.imshow(euler_com, cmap=mapname)
    plt.savefig('em-com.png')
    plt.close('all')

    euler_com_bm = ( g(zt) * rho_parabolic(mu2, mu3, s, B, dt, dB) ).mean(0)
    plt.imshow(euler_com_bm, cmap=mapname)
    plt.savefig('em-com_bm.png')
    plt.close('all')
    fig = plt.figure(figsize=(16,4.5))

    print('MSE {}'.format(((euler.mean(0) - biased)**2).mean()))

    vmax = biased.max()
    vmin = biased.min()

    plt.figure(figsize=(12,8))

    plt.subplot(121)
    plt.title('Original')
    plt.imshow(g(Bsig).mean(0), cmap=mapname, vmin=vmin, vmax=vmax, aspect='auto')
    x_low_bound  = 0
    x_high_bound = T
    y_low_bound  = 0.1
    y_high_bound = 1
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines['left'].set_bounds(0, Bsig.shape[-1])
    plt.gca().spines['left'].set_position(('outward', 5))
    plt.gca().spines['bottom'].set_bounds(0, Bsig.shape[-2])
    plt.gca().spines['bottom'].set_position(('outward', 5))
    plt.gca().set_xticks([0, Bsig.shape[-2]])
    plt.gca().set_yticks([0, Bsig.shape[-1]])
    plt.gca().set_xticklabels([x_low_bound, x_high_bound])
    plt.gca().set_yticklabels([y_low_bound, y_high_bound])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$X$')
    plt.tight_layout()
    plt.subplot(122)
    plt.title('Girsanov')
    plt.imshow(biased, cmap=mapname, vmin=vmin, vmax=vmax, aspect='auto')
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines['bottom'].set_bounds(0, Bsig.shape[-2])
    plt.gca().spines['bottom'].set_position(('outward', 5))
    plt.gca().set_xticks([0, Bsig.shape[-2]])
    plt.gca().set_xticklabels([x_low_bound, x_high_bound])
    plt.gca().set_yticks([])
    plt.gca().set_yticklabels([])
    plt.xlabel(r'$t$')
    plt.tight_layout()
    plt.savefig('com-sample-gbm.pdf')
    plt.close('all')

parabolic()
