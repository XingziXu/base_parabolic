import torch
from torch import nn
from torch.autograd import grad
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
#from torchvision import datasets, transforms
import numpy as np

from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform
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

D_IDX = -1
T_IDX = -2
N_IDX =  0

def drift(x, coef):
    return x

def diffusion(x,t):
    return 1

def initial(dim, x):
    pxt = torch.exp(-0.5*(x**2).sum(-1)) * (2*np.pi)**(-dim/2)
    return pxt

def r_value(dim, t):
    return torch.exp(- dim * t).to(device)

def ou_pdf(dim, x, t):
    t = t.to(device)
    pxt = torch.exp((x**2).sum(-1)/(1-3*torch.exp(2*t))) * (np.pi*(3*torch.exp(2*t)-1))**(-dim/2)
    return pxt

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(5, self.width) # input channel is 2: (a(x), x)
        self.act = nn.ReLU()

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.act(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.act(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.act(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

class CNN_expmart(nn.Module):
    def __init__(self, input_size, hidden_size, num_outputs):
        super(CNN_expmart, self).__init__()
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
    def __init__(self, N = 2000, lr = 1e-3, X = 1., T = 0.1, dim = 2, batch_size = 100, num_time = 100, n_batch_val=100):
        super().__init__()
        # define normalizing flow to model the conditional distribution rho(x,t)=p(y|x,t)
        self.num_time = num_time
        self.T = T
        self.t0 = 0.
        self.t = torch.linspace(0,1,steps=self.num_time)* self.T + self.t0
        self.dim = dim
        self.batch_size = batch_size
        # input size is dimension of brownian motion x 2, since the input to the RNN block is W_s^x and dW_s^x
        input_size = self.dim * 2 + 1
        # num_layers is the number of RNN blocks
        num_layers = 3
        # num_outputs is the number of ln(rho(x,t))
        num_outputs = self.dim
        self.expmart_cnn = CNN_expmart(input_size, 50 + self.dim * 5, num_outputs)
        self.model = FNO1d(modes=20, width=32)
        

        # define the learning rate
        self.lr = lr
        
        # define number of paths used and grid of PDE
        self.N = N
        self.dt = self.t[1]-self.t[0] # define time step
        #self.t = self.t + self.dt
        # define the brwonian motion starting at zero
        self.dB_em = np.sqrt(self.dt.item()) * np.random.randn(self.t.shape[0], self.N, self.batch_size, self.dim)
        self.dB_em[0,:,:,:] = 0 
        self.B0_em = self.dB_em.copy()
        self.B0_em = torch.Tensor(self.B0_em.cumsum(0)).to(device)
        self.dB_em = torch.Tensor(self.dB_em).to(device)
        
        self.dB_gir = np.sqrt(self.dt.item()) * np.random.randn(self.t.shape[0], self.N, self.batch_size, self.dim)
        self.dB_gir[0,:,:,:] = 0 
        self.B0_gir = self.dB_gir.copy()
        self.B0_gir = torch.Tensor(self.B0_gir.cumsum(0)).to(device)
        self.dB_gir = torch.Tensor(self.dB_gir).to(device)
        
        self.dB_cnn = np.sqrt(self.dt.item()) * np.random.randn(self.t.shape[0], self.N, self.batch_size, self.dim)
        self.dB_cnn[0,:,:,:] = 0 
        self.B0_cnn = self.dB_cnn.copy()
        self.B0_cnn = torch.Tensor(self.B0_cnn.cumsum(0)).to(device)
        self.dB_cnn = torch.Tensor(self.dB_cnn).to(device)
        
        self.cnn_metrics = torch.zeros((50,n_batch_val))
        self.gir_metrics = torch.zeros((50,n_batch_val))
        self.em_metrics = torch.zeros((50,n_batch_val))
        self.don_metrics = torch.zeros((50,n_batch_val))
        self.em_comp_time = torch.zeros((50,n_batch_val))
        self.gir_comp_time = torch.zeros((50,n_batch_val))
        self.cnn_comp_time = torch.zeros((50,n_batch_val))
        self.don_comp_time = torch.zeros((50,n_batch_val))
        self.epochs = torch.linspace(0,49,50)
        
        self.relu = torch.nn.Softplus()

        self.coef_train = torch.rand(10,1,1,3).to(device)
        
        self.relu = torch.nn.Softplus()
        
        self.m = 100 # number of "sensors" for the function u
        self.p = 15 # number of "branches"
        self.lr = 1e-30 # learning rate
        self.X = X # interval size
        self.sensors = initial(dim, (torch.linspace(0., 1., self.m).unsqueeze(-1).repeat(1,self.dim) * self.X).unsqueeze(0).unsqueeze(0).to(device))
        self.branch = MLP(input_dim=self.m, hidden_dim=70+5*dim, output_dim=self.p) # branch network
        self.trunk = MLP(input_dim=dim+1, hidden_dim=50+5*dim, output_dim=self.p) # trunk network

    def loss(self, xt, coef):
        coef = coef
        xs = torch.linspace(-1., 1., steps=self.batch_size).unsqueeze(-1).to(device)
        
        # calculate values using euler-maruyama
        start = time.time()
        x = torch.zeros(self.num_time, self.N, batch_size, self.dim).to(device)
        x[0,:,:,:] = xs
        for i in range(self.num_time-1):
            x[i+1,:,:,:] = x[i,:,:,:] + drift(x[i,:,:,:], coef) * self.dt + self.dB_em[i,:,:,:]
        p0mux = initial(dim, x).squeeze()
        u_em = (p0mux * r_value(self.dim, self.t).unsqueeze(-1).unsqueeze(-1)).mean(1)
        end = time.time()
        time_em = (end - start)
        
        # calculate values using girsanov
        Bx_gir = (xs.unsqueeze(0).unsqueeze(0)+self.B0_gir)
        p0Bx_gir = initial(dim, Bx_gir).squeeze()
        start = time.time()
        muBx_gir = drift(Bx_gir, coef)
        expmart = torch.exp((torch.cumsum(muBx_gir*self.dB_gir,dim=0) - 0.5 * torch.cumsum((muBx_gir ** 2) * self.dt,dim=0)).sum(-1))
        u_gir = (p0Bx_gir * expmart * r_value(self.dim, self.t).unsqueeze(-1).unsqueeze(-1)).mean(1)
        end = time.time()
        time_gir = (end - start)
        
        # calculate values using CNN
        Bx_cnn = (xs.unsqueeze(0).unsqueeze(0)+self.B0_cnn)
        p0Bx_cnn = initial(dim, Bx_cnn).squeeze()
        muBx_cnn = drift(Bx_cnn, coef)
        input = torch.zeros(self.num_time, self.N, self.batch_size, self.dim * 2 + 1).to(device)
        input[:muBx_cnn.shape[0],:,:,:] = torch.cat((muBx_cnn,self.dB_cnn,self.dt*torch.ones(self.dB_cnn.shape[0],self.dB_cnn.shape[1],self.dB_cnn.shape[2],1).to(device)),dim=-1)
        input_reshaped = input.reshape(input.shape[1]*input.shape[2], input.shape[3], input.shape[0])
        start = time.time()
        cnn_expmart = self.relu(self.expmart_cnn(input_reshaped).sum(-2)).reshape(p0Bx_cnn.shape)
        u_cnn = (p0Bx_cnn * cnn_expmart * r_value(self.dim, self.t).unsqueeze(-1).unsqueeze(-1)).mean(1)
        end = time.time()
        time_cnn = (end - start)
        
        # calculate ground truth values
        u_gt = ou_pdf(self.dim, xs.unsqueeze(0), self.t.unsqueeze(-1)).squeeze()
        
        # calculate values using deeponet
        axis = torch.Tensor([]).to(device)
        for i in range(self.num_time):
            current_axis = torch.cat((xs,(self.t0+i*self.dt)*torch.ones(xs.shape[0],1).to(device)),dim=1)
            axis = torch.cat((axis,current_axis),dim=0)
        start = time.time()
        branchs = self.branch(self.sensors.unsqueeze(0)).repeat(self.batch_size*(self.num_time), 1)
        u_don = torch.zeros_like(u_cnn)
        trunks = self.trunk(axis)
        u_don = (branchs * trunks).sum(1).reshape(u_gir.shape)
        end = time.time()
        time_don = (end - start)
        
        # calculate values using FNO
        start = time.time()
        ini_xs = initial(dim=1, x=xs)
        drift_xs = drift(xs, coef).squeeze()
        t_current = 0.
        t_channel = torch.ones_like(drift_xs) * t_current
        input = torch.cat((ini_xs.unsqueeze(-1), drift_xs.unsqueeze(-1), xs, t_channel.unsqueeze(-1)),dim=1).unsqueeze(0)
        u_fno = torch.zeros_like(u_gir)
        for ii in range(self.num_time):
            u_current = self.model(input)
            u_current = u_current.squeeze()
            u_fno[ii, :] = u_current
            t_current = self.dt * ii
            t_channel = torch.ones_like(drift_xs) * t_current
            input = torch.cat((u_current.unsqueeze(-1), drift_xs.unsqueeze(-1), xs, t_channel.unsqueeze(-1)),dim=1).unsqueeze(0)
            #input = torch.cat((u_current, drift_xs.unsqueeze(-1), xs.unsqueeze(-1)),dim=1).unsqueeze(0)
        end = time.time()
        
        time_cnn = (end - start)
        
        #with open('/scratch/xx84/girsanov/fk/ou/result/ou_cnn_1d_vis.npy', 'wb') as f:
        #    np.save(f, np.array(u_cnn.detach().cpu()))
        #with open('/scratch/xx84/girsanov/fk/ou/result/ou_gt_1d_vis.npy', 'wb') as f:
        #    np.save(f, np.array(u_gt.detach().cpu()))
        #with open('/scratch/xx84/girsanov/fk/ou/result/ou_don_1d_vis.npy', 'wb') as f:
        #    np.save(f, np.array(u_don.detach().cpu()))
        #with open('/scratch/xx84/girsanov/fk/ou/result/ou_gir_1d_vis.npy', 'wb') as f:
        #    np.save(f, np.array(u_gir.detach().cpu()))
        #with open('/scratch/xx84/girsanov/fk/ou/result/ou_em_1d_vis.npy', 'wb') as f:
        #    np.save(f, np.array(u_em.detach().cpu()))
        with open('/scratch/xx84/girsanov/fk/ou/result/ou_fno_1d_vis.npy', 'wb') as f:
            np.save(f, np.array(u_fno.detach().cpu()))
        return u_cnn, time_cnn, u_gt, u_fno, u_em

    def training_step(self, batch, batch_idx):
        # REQUIRED
        xt = batch.to(device)
        #xs = torch.linspace(-0.1, 0.1, batch_size)
        #t = torch.zeros_like(xs)
        #xt = torch.cat((xs.unsqueeze(-1),t.unsqueeze(-1)),dim=-1).to(device)
        u_cnn, time_cnn, u_gt, u_fno, u_em = self.loss(xt, coef=torch.rand(1,1,1,3).to(device))
        #plt.subplot(3,2,1)
        #plt.imshow(u_gt.cpu().detach().numpy())
        #plt.colorbar()
        #plt.subplot(3,2,2)
        #plt.imshow((u_gt-u_gt).cpu().detach().numpy())
        #plt.colorbar()
        #plt.subplot(3,2,3)
        #plt.imshow(u_em.cpu().detach().numpy())
        #plt.colorbar()
        #plt.subplot(3,2,4)
        #plt.imshow((u_em-u_gt).cpu().detach().numpy())
        #plt.colorbar()
        #plt.subplot(3,2,5)
        #plt.imshow(u_gir.cpu().detach().numpy())
        #plt.colorbar()
        #plt.subplot(3,2,6)
        #plt.imshow((u_gir-u_gt).cpu().detach().numpy())
        #plt.colorbar()
        #plt.savefig('/scratch/xx84/girsanov/pde_rnn/ou_ground_truth.png')
        loss = F.l1_loss(u_fno, u_em)#/(torch.abs(u_gir).mean())
        #tensorboard_logs = {'train_loss': loss_prior}
        self.log('train_loss', loss)
        #print(loss_total)
        return {'loss': loss}
        
        
    def validation_step(self, batch, batch_idx):
        #xs = torch.linspace(-0.1, 0.1, batch_size)
        #t = torch.zeros_like(xs)
        #xt = torch.cat((xs.unsqueeze(-1),t.unsqueeze(-1)),dim=-1).to(device)
        xt = batch.to(device)
        self.expmart_cnn.load_state_dict(torch.load('/scratch/xx84/girsanov/fk/ablation/trained_model/ngo_'+str(self.dim)+'.pt'))
        self.branch.load_state_dict(torch.load('/scratch/xx84/girsanov/fk/ablation/trained_model/branch_'+str(self.dim)+'.pt'))
        self.trunk.load_state_dict(torch.load('/scratch/xx84/girsanov/fk/ablation/trained_model/trunk_'+str(self.dim)+'.pt'))
        self.model.load_state_dict(torch.load('/scratch/xx84/girsanov/fk/high_dim/trained_model/tfno_1.pt'))
        u_cnn, time_cnn, u_gt, u_fno, u_em = self.loss(xt, coef=torch.rand(1,1,1,3).to(device))
        
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
    em_loss_min = []
    em_loss_mean = []
    em_loss_max = []
    em_loss_var = []
    
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
    
    i=100
    dim = 1
    X = 0.5
    T = i * 0.01
    num_time = i
    num_samples = 200
    batch_size = 100
    N = 3000
    xs = torch.rand(num_samples,dim) * X
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

    model = FKModule(X=X, T=T, batch_size=batch_size, dim=dim, num_time=num_time, N=N, n_batch_val=n_batch_val)
    trainer = pl.Trainer(max_epochs=1, gpus=1, check_val_every_n_epoch=1)
    trainer.fit(model, train_loader, val_loader)