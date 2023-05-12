import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib
import seaborn as sns

matplotlib.rcParams.update({'font.size': 15})

with open('/scratch/xx84/girsanov/generative_modeling/fk/result/100dgaussian_bpd_nf_0.npy', 'rb') as f:
    bpd_nf_100 = np.load(f)
with open('/scratch/xx84/girsanov/generative_modeling/fk/result/100dgaussian_loss_nf_0.npy', 'rb') as f:
    wass_nf_100 = np.load(f)

with open('/scratch/xx84/girsanov/generative_modeling/fk/result/100dgaussian_bpd_std_0.npy', 'rb') as f:
    bpd_std_100 = np.load(f)
with open('/scratch/xx84/girsanov/generative_modeling/fk/result/100dgaussian_loss_std_0.npy', 'rb') as f:
    wass_std_100 = np.load(f)
    
with open('/scratch/xx84/girsanov/generative_modeling/fk/result/2dgaussian_bpd_nf_0.npy', 'rb') as f:
    bpd_nf_2 = np.load(f)
with open('/scratch/xx84/girsanov/generative_modeling/fk/result/2dgaussian_loss_nf_0.npy', 'rb') as f:
    wass_nf_2 = np.load(f)

with open('/scratch/xx84/girsanov/generative_modeling/fk/result/2dgaussian_bpd_std_0.npy', 'rb') as f:
    bpd_std_2 = np.load(f)
with open('/scratch/xx84/girsanov/generative_modeling/fk/result/2dgaussian_loss_std_0.npy', 'rb') as f:
    wass_std_2 = np.load(f)
#plt.set(xscale="linear", yscale="symlog")
#plt.xscale('symlog')
#plt.yscale('symlog')

#sns.kdeplot(x=bpd_nf, y=loss_nf, thresh=.2, label='$p_{meta}$')
#sns.kdeplot(x=bpd_std, y=loss_std, thresh=.2, label='Gaussian')

max_wass_2 = np.max(wass_nf_2)
bpd_std_2 = bpd_std_2[(wass_std_2 <= max_wass_2)]
wass_std_2 = wass_std_2[(wass_std_2 <= max_wass_2)]

max_wass_100 = np.max(wass_nf_100)
bpd_std_100 = bpd_std_100[(wass_std_100 <= max_wass_100)]
wass_std_100 = wass_std_100[(wass_std_100 <= max_wass_100)]

f = plt.figure(figsize=(12, 8))
gs = f.add_gridspec(1, 2)

with sns.axes_style("darkgrid"):
    ax = f.add_subplot(gs[0])
    sns.regplot(x=wass_nf_2, y=bpd_nf_2, scatter_kws={'color': 'purple', 'alpha': 0.3}, line_kws = {'color': 'purple', 'alpha': 0.3, 'lw':6}, label='$p_{meta}$')
    sns.regplot(x=wass_std_2, y=bpd_std_2, scatter_kws={'color': 'green', 'alpha': 0.3}, line_kws={'color': 'green'}, label='Gaussian', x_ci = 99)
    ax.set(xlabel='Wasserstein Distance', ylabel='Bits/Dim', title='2D')

with sns.axes_style("darkgrid"):
    ax = f.add_subplot(gs[1])
    sns.regplot(x=wass_nf_100, y=bpd_nf_100, scatter_kws = {'color': 'purple', 'alpha': 0.3}, line_kws = {'color': 'purple', 'alpha': 0.3, 'lw':6}, label='$p_{meta}$')
    sns.regplot(x=wass_std_100, y=bpd_std_100, scatter_kws={'color': 'green', 'alpha': 0.3}, line_kws={'color': 'green'}, label='Gaussian', x_ci = 99)
    ax.set(xlabel='Wasserstein Distance', ylabel='Bits/Dim', title='100D')

handles, labels = ax.get_legend_handles_labels()
f.legend(handles, labels, loc='upper right')

plt.tight_layout()
plt.savefig('/scratch/xx84/girsanov/generative_modeling/fk/figure/meta_reg.pdf')