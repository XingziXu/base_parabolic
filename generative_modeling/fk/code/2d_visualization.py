import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib
import seaborn as sns

matplotlib.rcParams.update({'font.size': 15})

with open('/scratch/xx84/girsanov/generative_modeling/fk/result/2dgaussian_bpd_nf_0.npy', 'rb') as f:
    bpd_nf = np.load(f)
with open('/scratch/xx84/girsanov/generative_modeling/fk/result/2dgaussian_loss_nf_0.npy', 'rb') as f:
    loss_nf = np.load(f)

with open('/scratch/xx84/girsanov/generative_modeling/fk/result/2dgaussian_bpd_std_0.npy', 'rb') as f:
    bpd_std = np.load(f)
with open('/scratch/xx84/girsanov/generative_modeling/fk/result/2dgaussian_loss_std_0.npy', 'rb') as f:
    loss_std = np.load(f)
    
#plt.set(xscale="linear", yscale="symlog")
#plt.xscale('symlog')
#plt.yscale('symlog')

#sns.kdeplot(x=bpd_nf, y=loss_nf, thresh=.2, label='$p_{meta}$')
#sns.kdeplot(x=bpd_std, y=loss_std, thresh=.2, label='Gaussian')

sns.regplot(x=bpd_nf, y=loss_nf, scatter_kws={"alpha" : 0.2}, label='$p_{meta}$', color='royalblue')
sns.regplot(x=bpd_std, y=loss_std, scatter_kws={"alpha" : 0.2}, label='Gaussian', color='orangered')
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines['left'].set_position(('outward', 5))
plt.gca().spines['bottom'].set_position(('outward', 5))
plt.xlabel('Bits/Dim')
plt.ylabel('Validation Loss')
plt.legend()
plt.xlim(min(bpd_nf),max(bpd_nf))
plt.tight_layout()
plt.savefig('/scratch/xx84/girsanov/generative_modeling/fk/figure/2d_kde.png')