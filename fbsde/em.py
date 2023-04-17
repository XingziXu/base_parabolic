# simulation based on section 5 of https://www.sciencedirect.com/science/article/pii/S0005109817304740

import torch
from torch import nn
from torch.autograd import grad
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import pytorch_lightning as pl
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
from random import randint
import seaborn as sns 
import pandas as pd
from torchqrnn import QRNN
import time

def b(t,x):
    return torch.sin(x)

def sigma(t,x):
    return torch.cos(t)

def g(x):
    return torch.sin(x)

def h(t,x,y,z):
    return torch.sin(t)+torch.cos(x)+(y ** 2)+z



if __name__ == '__main__':
    pl.seed_everything(1234)
    print(sys.executable)
    #dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    #mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
    #mnist_train, mnist_val = random_split(dataset, [55000,5000])
    device = torch.device("cuda:0")
    
    x_num = 20
    x0 = torch.randn()