from gir_gen import FKModule
from nets import SMLP

import torch
import torch.nn as nn

from torchvision.datasets import USPS
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, TensorDataset

from pytorch_lightning import Trainer, seed_everything

from sklearn.datasets import make_moons

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

d = 2
EPOCHS = 1000
bs = 20

trainer_params = {
        'gpus' : 1,
        'check_val_every_n_epoch' : 1,
        'max_epochs' : EPOCHS,
        #'progress_bar_refresh_rate' : 0,
        #'checkpoint_callback' : False,
        #'logger' : False
        }
datatype = 'moons'
#dataset = USPS(root='../../Data/USPS', train=True, download=True, transform=transforms.ToTensor())

if datatype == 'moons':
    mx, my = make_moons(n_samples=500, noise=0.05)
elif datatype == 'twogauss':
    a = torch.randn(250,2)*0.1 - 0.5
    b = torch.randn(250,2)*0.1 + 0.5
    mx = torch.cat((a,b))
elif datatype == 'circle':
    uni = 2* torch.pi * torch.rand(500,1)
    a = torch.sin(uni)
    b = torch.cos(uni)
    mx = torch.cat((a,b),1)

dataset = TensorDataset(torch.Tensor(mx), torch.Tensor(my))
train_loader = DataLoader(dataset, batch_size=bs,
                        shuffle=True, num_workers=0)
# fix this
print('train and validation are the same')
val_loader = DataLoader(dataset, batch_size=500,
                        shuffle=True, num_workers=0)

plt.scatter(mx[:,0], mx[:,1])
plt.savefig('orig.pdf')
plt.close('all')

# drift
mu = SMLP(d, 200, 3, d).to('cuda:0')
#nn.init.zeros_(mu.out.weight.data)

# boundary condition
mn = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2).to('cuda:0'), torch.eye(2).to('cuda:0'))
g = lambda x: mn.log_prob(x).exp()

fkm = FKModule(d, mu, g, lr_mu=1e-3)

trainer = Trainer(**trainer_params)
trainer.fit(fkm, train_loader, val_loader)