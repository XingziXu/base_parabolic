import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib
import seaborn as sns

matplotlib.rcParams.update({'font.size': 35})

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
    
bpd_mu_2 = np.load('/scratch/xx84/girsanov/generative_modeling/fk/result/2dgaussian_bpd_mu_0.npy')
wass_mu_2 = np.load('/scratch/xx84/girsanov/generative_modeling/fk/result/2dgaussian_loss_mu_0.npy')
mag_mu_2 = np.load('/scratch/xx84/girsanov/generative_modeling/fk/result/2dgaussian_mag_mu_0.npy')

bpd_mu_100 = np.load('/scratch/xx84/girsanov/generative_modeling/fk/result/100dgaussian_bpd_mu_0.npy')
wass_mu_100 = np.load('/scratch/xx84/girsanov/generative_modeling/fk/result/100dgaussian_loss_mu_0.npy')
mag_mu_100 = np.load('/scratch/xx84/girsanov/generative_modeling/fk/result/100dgaussian_mag_mu_0.npy')
#wass_nf_2 = np.load('/scratch/xx84/girsanov/generative_modeling/fk/result/2dgaussian_bpd_nf_0.npy')
#loss_nf_2 = np.load('/scratch/xx84/girsanov/generative_modeling/fk/result/2dgaussian_loss_nf_0.npy')
#plt.scatter(loss_nf_2, val_nf_2, label='Prior')
#plt.scatter(loss_mu_2, val_mu_2, label='Prior+Drift')
#plt.savefig('/scratch/xx84/girsanov/generative_modeling/fk/figure/scratch.png')

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

with sns.axes_style("whitegrid"):
    ax1 = f.add_subplot(gs[0])
    sns.regplot(x=wass_std_2, y=bpd_std_2, scatter_kws={'color': 'tomato', 'alpha': 0.3, 's':100}, line_kws={'color': 'tomato', 'alpha': 0.8,'lw':2})
    legend_line_1 = plt.scatter( np.NaN, np.NaN, s = 100, color='tomato', alpha=0.7, label='Gaussian')
    sns.regplot(x=wass_nf_2, y=bpd_nf_2, scatter_kws={'color': 'teal', 'alpha': 0.3, 's':100}, line_kws = {'color': 'teal', 'alpha': 0.8,'lw':2})
    legend_line_2 = plt.scatter( np.NaN, np.NaN, s = 100, color='teal', alpha=0.7, label='$p_{meta}$')
    sns.regplot(x=wass_mu_2, y=bpd_mu_2, scatter_kws={'color': 'darkorange', 'alpha': 0.3, 's':100}, line_kws = {'color': 'darkorange', 'alpha': 0.8,'lw':2})
    legend_line_3 = plt.scatter( np.NaN, np.NaN, s = 100, color='darkorange', alpha=0.7, label='$p_{meta}+\mu_{meta}$')
    ax1.set_xlim(min(wass_std_2)-10, max(wass_std_2)+10)
    ax1.set(xlabel='Wass Distance', ylabel='Bits/Dim')

    ax2 = f.add_subplot(gs[1])
    sns.regplot(x=wass_std_100, y=bpd_std_100, scatter_kws={'color': 'tomato', 'alpha': 0.3, 's':100}, line_kws = {'color': 'tomato', 'alpha': 0.9,'lw':2}, label='Gaussian')
    sns.regplot(x=wass_nf_100, y=bpd_nf_100, scatter_kws={'color': 'teal', 'alpha': 0.3, 's':100}, line_kws = {'color': 'teal', 'alpha': 0.9,'lw':2}, label='$p_{meta}$')
    sns.regplot(x=wass_mu_100, y=bpd_mu_100, scatter_kws={'color': 'darkorange', 'alpha': 0.3, 's':100}, line_kws = {'color': 'darkorange', 'alpha': 0.9,'lw':2}, label='$p_{meta}+\mu_{meta}$')
    ax2.set_xlim(min(wass_std_100)-10, max(wass_std_100)+10)
    ax2.set(xlabel='Wass Distance')
    
#with sns.axes_style("whitegrid"):
#    ax = f.add_subplot(gs[1])
#    sns.regplot(x=wass_nf_100, y=bpd_nf_100, scatter_kws = {'color': 'teal', 'alpha': 0.3, 's':100}, line_kws = {'color': 'teal', 'alpha': 0.9,'lw':2}, label='$p_{meta}$')
#    sns.regplot(x=wass_std_100, y=bpd_std_100, scatter_kws={'color': 'tomato', 'alpha': 0.3, 's':100}, line_kws={'color': 'tomato', 'alpha': 0.9,'lw':2}, label='Gaussian')
#    ax.set(xlabel='Wasserstein Distance', title='$100$-$d$')

handles, labels = ax1.get_legend_handles_labels()
f.legend(handles, labels, loc='upper center')

plt.tight_layout()
plt.savefig('/scratch/xx84/girsanov/generative_modeling/fk/figure/meta_reg.pdf')
plt.clf()
