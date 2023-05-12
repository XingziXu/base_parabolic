import torch
import numpy as np
import normflows as nf

from sklearn.datasets import make_moons
from torch.distributions import MultivariateNormal, Uniform, TransformedDistribution, SigmoidTransform

from matplotlib import pyplot as plt

from tqdm import tqdm
from torch import nn

# Set up model

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

dim = 100
# Define 2D Gaussian base distribution
base = nf.distributions.base.DiagGaussian(dim)

# Define list of flows
num_layers = 32
flows = []
for i in range(num_layers):
    # Neural network with two hidden layers having 64 units each
    # Last layer is initialized by zeros making training more stable
    #param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
    param_map = SMLP(int(dim/2), 128, 3, dim, nn.Tanh())
    # Add flow layer
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    # Swap dimensions
    flows.append(nf.flows.Permute(2, mode='swap'))
    
# Construct flow model
model = nf.NormalizingFlow(base, flows)

# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
model = model.to(device)

"""
# Plot target distribution
x_np, _ = make_moons(2 ** 20, noise=0.1)
plt.figure(figsize=(15, 15))
plt.hist2d(x_np[:, 0], x_np[:, 1], bins=200, range=[[-1.5, 2.5], [-2, 2]])
plt.show()

# Plot initial flow distribution
grid_size = 100
xx, yy = torch.meshgrid(torch.linspace(-1.5, 2.5, grid_size), torch.linspace(-2, 2, grid_size))
zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
zz = zz.to(device)

model.eval()
log_prob = model.log_prob(zz).to('cpu').view(*xx.shape)
model.train()
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

plt.figure(figsize=(15, 15))
plt.pcolormesh(xx, yy, prob.data.numpy())
plt.gca().set_aspect('equal', 'box')
plt.savefig('/scratch/xx84/girsanov/generative_modeling/nf/figure/initial.png')
plt.clf()
"""

# Train model
max_iter = 1000
num_samples = 600
show_iter = 500


loss_hist = np.array([])

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

p0 = MultivariateNormal(torch.ones(dim) * 0.0, torch.eye(dim))
p1 = MultivariateNormal(torch.ones(dim) * 1.0, torch.eye(dim))
p2 = MultivariateNormal(torch.ones(dim) * 2.0, torch.eye(dim))
p3 = MultivariateNormal(torch.ones(dim) * 3.0, torch.eye(dim))
p4 = MultivariateNormal(torch.ones(dim) * 4.0, torch.eye(dim))
p5 = MultivariateNormal(torch.ones(dim) * 5.0, torch.eye(dim))
p6 = MultivariateNormal(torch.ones(dim) * 6.0, torch.eye(dim))
p7 = MultivariateNormal(torch.ones(dim) * 7.0, torch.eye(dim))
p8 = MultivariateNormal(torch.ones(dim) * 8.0, torch.eye(dim))
p9 = MultivariateNormal(torch.ones(dim) * 9.0, torch.eye(dim))

for it in tqdm(range(max_iter)):
    optimizer.zero_grad()
    
    # Get training samples
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
    
    x_np = np.concatenate((x0,x1,x2,x3,x4,x5,x6,x7,x8,x9),axis=0)
    x = torch.tensor(x_np).float().to(device)
    
    # Compute loss
    loss = model.forward_kld(x)
    
    # Do backprop and optimizer step
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()
    
    # Log loss
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
    # Plot learned posterior
    """
    if (it + 1) % show_iter == 0:
        model.eval()
        log_prob = model.log_prob(zz)
        model.train()
        prob = torch.exp(log_prob.to('cpu').view(*xx.shape))
        prob[torch.isnan(prob)] = 0

        plt.figure(figsize=(15, 15))
        plt.pcolormesh(xx, yy, prob.data.numpy())
        plt.gca().set_aspect('equal', 'box')
        plt.savefig('/scratch/xx84/girsanov/generative_modeling/nf/figure/posterior.png')
        plt.clf()
    """
# Plot loss
plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.savefig('/scratch/xx84/girsanov/generative_modeling/nf/figure/loss_100.png')

"""
# Plot learned posterior distribution
model.eval()
log_prob = model.log_prob(zz).to('cpu').view(*xx.shape)
model.train()
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

plt.figure(figsize=(15, 15))
plt.pcolormesh(xx, yy, prob.data.numpy())
plt.gca().set_aspect('equal', 'box')
plt.savefig('/scratch/xx84/girsanov/generative_modeling/nf/figure/posterior_final.png')
"""
torch.save(model.state_dict(), '/scratch/xx84/girsanov/generative_modeling/nf/trained_model/nf_prior_100.pt')