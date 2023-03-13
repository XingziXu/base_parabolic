import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def init_weights(net, init_dict, gain=1, input_class=None):
    def init_func(m):
        if input_class is None or type(m) == input_class:
            for key, value in init_dict.items():
                param = getattr(m, key, None)
                if param is not None:
                    if value == 'normal':
                        nn.init.normal_(param.data, 0.0, gain)
                    elif value == 'xavier':
                        nn.init.xavier_normal_(param.data, gain=gain)
                    elif value == 'kaiming':
                        nn.init.kaiming_normal_(param.data, a=0, mode='fan_in')
                    elif value == 'orthogonal':
                        nn.init.orthogonal_(param.data, gain=gain)
                    elif value == 'uniform':
                        nn.init.uniform_(param.data)
                    elif value == 'zeros':
                        nn.init.zeros_(param.data)
                    elif value == 'very_small':
                        nn.init.constant_(param.data, 1e-3*gain)
                    elif value == 'ones':
                        nn.init.constant_(param.data, 1)
                    elif value == 'xavier1D':
                        nn.init.normal_(param.data, 0.0, gain/param.numel().sqrt())
                    elif value == 'identity':
                        nn.init.eye_(param.data)
                    else:
                        raise NotImplementedError('initialization method [%s] is not implemented' % value)
    net.apply(init_func)

#activation functions
class quadratic(nn.Module):
    def __init__(self):
        super(quadratic,self).__init__()

    def forward(self,x):
        return x**2

class quadratic(nn.Module):
    def __init__(self):
        super(quadratic,self).__init__()

    def forward(self,x):
        return x*F.relu(x)

class cos(nn.Module):
    def __init__(self):
        super(cos,self).__init__()

    def forward(self,x):
        return torch.cos(x)

class sin(nn.Module):
    def __init__(self):
        super(sin,self).__init__()

    def forward(self,x):
        return torch.sin(x)

class swish(nn.Module):
    def __init__(self):
        super(swish,self).__init__()

    def forward(self,x):
        return torch.sigmoid(x)*x

class relu2(nn.Module):
    def __init__(self,order=2):
        super(relu2,self).__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.order = order

    def forward(self,x):
        return F.relu(x)**(self.order)

class leakyrelu2(nn.Module):
    def __init__(self,order=2):
        super(leakyrelu2,self).__init__()
        self.a = nn.Parameter(torch.ones(1))
        #self.a = torch.ones(1)
        self.order = order

    def forward(self,x):
        return F.leaky_relu(self.a.to(x.device)*x)**self.order

class mod_softplus(nn.Module):
    def __init__(self):
        super(mod_softplus,self).__init__()

    def forward(self,x):
        return F.softplus(x) + x/2 - torch.log(torch.ones(1)*2).to(device=x.device)

class mod_softplus2(nn.Module):
    def __init__(self):
        super(mod_softplus2,self).__init__()

    def forward(self,x,d):
        return d*(1+d)*(2*F.softplus(x) - x  - 2*torch.log(torch.ones(1)*2).to(device=x.device))

class mod_softplus3(nn.Module):
    def __init__(self):
        super(mod_softplus3,self).__init__()

    def forward(self,x):
        return F.relu(x) + F.softplus(-torch.abs(x)) 

class swish(nn.Module):
    def __init__(self):
        super(swish,self).__init__()

    def forward(self,x):
        return x*torch.sigmoid(x) 

class soft2(nn.Module):
    def __init__(self):
        super(soft2,self).__init__()

    def forward(self,x):
        return torch.sqrt(x**2 + 1) / 2 + x / 2

class soft3(nn.Module):
    def __init__(self):
        super(soft3,self).__init__()

    def forward(self,x):
        return torch.logsigmoid(-x) 

class Shallow(nn.Module):
    def __init__(self,input_size,out_size):
        super(Shallow, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_size,input_size),quadratic(),nn.Linear(input_size,out_size))

    def forward(self,x):
        return self.net(x)

class PositiveLinear(nn.Linear):
    def __init__(self, **args):
        super(PositiveLinear, self).__init__()


class SMLP(nn.Module):
    def __init__(self, input_size, hidden_size, layers, out_size, act=nn.CELU()):
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

class ICNN_old(nn.Module):
    def __init__(self, input_size, width, depth, out_size, fn=nn.Tanh()):
        super(ICNN, self).__init__()

        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fn = fn
        mid_list_z = [nn.Linear(hidden_size,hidden_size)]
        mid_list_x = [nn.Linear(input_size,hidden_size)]
        for i in range(layers-1):
           mid_list_z += [nn.Linear(hidden_size, hidden_size)]
           mid_list_x += [nn.Linear(input_size, hidden_size)]
        self.out_x = nn.Linear(hidden_size, out_size)
        self.out_z = nn.Linear(input_size, out_size)

    def forward(self, x):
        z1 = self.fc1(x)
        wzi = z1
        for xlayer in mid_list_x:
            wzi = zlayer(zi)
            wxi = xlayer(x)
            zi = self.fn(wxi + wzi)
        out = self.out_x(x) + self.out_z(z)

class PositionalEncodingLayer(nn.Module):
    def __init__(self, L=20):
        super(PositionalEncodingLayer, self).__init__()
        scale1 = 2**torch.arange(0, L)*math.pi
        scale2 = 2**torch.arange(0, L)*math.pi + math.pi 
        self.scale = torch.stack((scale1,scale2),1).view(1,-1).to('cuda:0')

    def forward(self, x):
        return torch.sin(x.unsqueeze(-1) @ self.scale).view(x.shape[0], x.shape[1], x.shape[2],-1)

class SqrtReLUBeta(nn.Module):
    def __init__(self):
        super(SqrtReLUBeta, self).__init__()
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return (torch.sqrt(self.beta.to(x.device)**2 + x ** 2) + x) / 2

class MaxElementMult(nn.Module):
    def __init__(self, in_size, out_size):
        super(MaxElementMult, self).__init__()
        a = torch.rand(out_size, in_size)#.round()
        self.in_size = in_size
        self.W = nn.Parameter(a)

    def forward(self, x):
        return torch.max(self.W * x.unsqueeze(1), -1)[0]

    def clamp(self):
        self.W.data.clamp_(0)

class MaxLinear(nn.Module):
    def __init__(self, input_size, width, depth, cond_size=0, cond_width=0):
        super(MaxLinear, self).__init__()

        layers = []

        self.W0 = MaxElementMult(input_size, width)

        for _ in range(depth-1):
            Wi = MaxElementMult(width, width)
            layers.append(Wi)

        self.Wi = nn.Sequential(*layers)

    def forward(self, x, cond=None):
        assert len(x.shape) == 2, 'We will unsqueeze it here.'

        out = self.W0(x)
        out = self.Wi(out)

        return out.mean(-1, keepdims=True)

    def clamp(self):
        self.W0.clamp()
        for block in self.Wi:
            block.clamp()

class ICNN(nn.Module):
    def __init__(self, input_size, width, depth, 
            cond_size=0, 
            cond_width=0, 
            fn0=relu2(order=2), 
            fn=nn.LeakyReLU(), 
            fnu=nn.LeakyReLU()):

        super(ICNN, self).__init__()

        self.fn0 = fn0
        self.fn = fn
        self.cond_size = cond_size

        self.fc0 = nn.Linear(input_size,width,bias=True)

        if cond_size > 0:
            self.uc0   = nn.Linear(cond_size, cond_width, bias=True)
            self.cc0   = nn.Linear(cond_size, width, bias=False)
            mid_list   = [PICNN_block(input_size,width,width,cond_width,fn,fnu) for i in range(depth-1)]
            mid_list.append(PICNN_block(input_size,width,1,cond_width,nn.Softplus(),fnu))
        else:
            mid_list = [ICNN_block(input_size,width,fn) for i in range(depth-1)]
            self.out_z = nn.Linear(width, 1, bias=False)
            self.out_x = nn.Linear(input_size, 1, bias=True)

        self.mid = nn.Sequential(*mid_list)
        init_weights(self, {'weight': 'orthogonal', 'bias': 'zeros'}, gain=1)

    def forward(self, x, cond=None):
        z0 = self.fc0(x)

        if self.cond_size > 0:
            u0 = self.uc0(cond)
            c0 = self.cc0(cond)
            z0 = self.fn0(z0 + c0)
            _, z, _ = self.mid((x, z0, u0))
            return z
        else:
            z0 = self.fn0(z0)
            _, z = self.mid((x,z0))
            out = (self.out_x(x) + self.out_z(z))
            return out

    def clamp(self):
        if self.cond_size == 0:
            self.out_z.weight.data.clamp_(0)
        for block in self.mid:
            block.clamp()

class ICNN_block(nn.Module):
    def __init__(self, x_size, zi_size, fn):
        super(ICNN_block, self).__init__()
        self.lin_x = nn.Linear(x_size, zi_size, bias=True)
        self.lin_z = nn.Linear(zi_size, zi_size, bias=False)
        self.fn = fn

    def forward(self, input_):
        x = input_[0]
        z = input_[1]
        out = self.fn(self.lin_x(x) + self.lin_z(z))
        return (x, out)

    def clamp(self):
        self.lin_z.weight.data.clamp_(0)


class PICNN_block(nn.Module):
    def __init__(self, x_size, zi_size, zout_size, ui_size, fn, fnu):
        super(PICNN_block, self).__init__()

        self.lin_u_hat = nn.Linear(ui_size, ui_size, bias=True)

        self.lin_u  = nn.Linear(ui_size, zout_size, bias=True)
        self.lin_uz = nn.Linear(ui_size, zi_size, bias=True)
        self.lin_ux = nn.Linear(ui_size, x_size,  bias=True)

        self.lin_x  = nn.Linear(x_size,  zout_size, bias=False)
        self.lin_z  = nn.Linear(zi_size, zout_size, bias=False)

        self.fn  = fn
        self.fnu = fnu

        #nn.init.ones_(self.lin_uz.bias.data)
        #nn.init.ones_(self.lin_ux.bias.data)

    def forward(self, input_):

        x = input_[0]
        z = input_[1]
        u = input_[2]

        u1  = self.fnu( self.lin_u_hat( u ) ) 

        pos = self.lin_z( z * F.relu( self.lin_uz( u ) ) )
        wx  = self.lin_x( x * self.lin_ux( u ) )
        wu  = self.lin_u( u )
        z1 = pos + wx + wu

        if self.fn:
            z1  = self.fn( z1 ) 

        return (x, z1, u1)

    def clamp(self):
        self.lin_z.weight.data.clamp_(0)

class L2Proj(nn.Module):
    def __init__ (self):
        super(L2Proj, self).__init__()
    def forward(self, x):
        if torch.norm(x) > 1:
            return x/torch.norm(x)
        else:
            return x

def siren_init(m):
    if type(m) == nn.Linear:
        c = 6
        n = m.weight.shape[0]
        torch.nn.init.uniform_(m.weight, -np.sqrt(c/n), np.sqrt(c/n))

class MLP_sin(nn.Module):
    def __init__(self, input_size, hidden_size, layers, out_size, n=10, proj=False, bn=False):
        super(MLP_sin, self).__init__()
        fn = sin()

        self.fc1 = nn.Linear(input_size,hidden_size,bias=True)
        self.layers = layers
        if layers > 0:
            mid_list = [nn.Linear(hidden_size,hidden_size,bias=False) for _ in range(layers)]
            self.mid = nn.Sequential(*mid_list)
        self.out = nn.Linear(hidden_size, out_size)
        self.w0 = 30
        c = 6
        n = hidden_size
        nn.init.uniform_(self.fc1.weight, -np.sqrt(c/self.w0/input_size), np.sqrt(c/self.w0/input_size))
        self.mid.apply(siren_init)
        nn.init.uniform_(self.out.weight, -np.sqrt(c/n), np.sqrt(c/n))
        
    def forward(self,x):
        out = self.fc1(30*x)
        if self.layers > 0:
            out = self.mid(out)
        out = self.out(out)
        return out