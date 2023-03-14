import matplotlib.pyplot as plt
import torch
import numpy as np

loss_std = np.load('/scratch/xx84/girsanov/generative_modeling/2dgaussian_loss_std_0.npy')

loss_meta = np.load('/scratch/xx84/girsanov/generative_modeling/2dgaussian_loss_nf_0.npy')

bpd_std = np.load('/scratch/xx84/girsanov/generative_modeling/2dgaussian_bpd_std_0.npy')

bpd_meta = np.load('/scratch/xx84/girsanov/generative_modeling/2dgaussian_bpd_nf_0.npy')

plt.scatter(loss_std, bpd_std, c='r', label='std')
plt.scatter(loss_meta, bpd_meta, c='b', label='meta')
plt.xlabel('Wasserstein Distance')
plt.ylabel('ELBO Loss')
plt.legend()
plt.savefig('/scratch/xx84/girsanov/visualized.png')