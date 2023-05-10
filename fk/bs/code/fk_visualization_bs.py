import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib

matplotlib.rcParams.update({'font.size': 15})

with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_cnn_loss_mean.npy', 'rb') as f:
    cnn_mean = np.load(f)
with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_cnn_loss_min.npy', 'rb') as f:
    cnn_min = np.load(f)
with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_cnn_loss_max.npy', 'rb') as f:
    cnn_max = np.load(f)
with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_em_loss_mean.npy', 'rb') as f:
    em_mean = np.load(f)
with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_em_loss_min.npy', 'rb') as f:
    em_min = np.load(f)
with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_em_loss_max.npy', 'rb') as f:
    em_max = np.load(f)
with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_don_loss_mean.npy', 'rb') as f:
    don_mean = np.load(f)
with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_don_loss_min.npy', 'rb') as f:
    don_min = np.load(f)
with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_don_loss_max.npy', 'rb') as f:
    don_max = np.load(f)
#with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_gir_loss_mean.npy', 'rb') as f:
#    gir_mean = np.load(f)
#with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_gir_loss_min.npy', 'rb') as f:
#    gir_min = np.load(f)
#with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_gir_loss_max.npy', 'rb') as f:
#    gir_max = np.load(f)


ep = torch.arange(len(cnn_mean)) * 0.025

#plt.plot(ep, np.array(gir_mean), label='Girsanov', color='palevioletred')
#plt.fill_between(ep, np.array(gir_min), np.array(gir_max), alpha=0.2, color='lightpink')
ax = plt.subplot(111)
ax.plot(ep, np.array(cnn_mean), label='NGO', color='mediumturquoise')
ax.fill_between(ep, np.array(cnn_min), np.array(cnn_max), alpha=0.2, color='mediumturquoise')
ax.set_ylim(0, np.array(don_max).max()+0.2)
ax.plot(ep, np.array(don_mean), label='DeepONet',color='slateblue')
ax.fill_between(ep, np.array(don_min), np.array(don_max), alpha=0.2,color='slateblue')
ax.plot(ep, np.array(em_mean), label='Euler-Maruyama',color='tomato')
ax.fill_between(ep, np.array(em_min), np.array(em_max), alpha=0.2,color='tomato')
ax.grid()
ax.set_ylabel('Normalized Error')
ax.set_xlabel('Terminal Time')
ax.legend()
ax.plot(ep, np.array(cnn_mean), label='NGO', color='white')
ax.plot(ep, np.array(don_mean), label='DeepONet',color='white')
ax.plot(ep, np.array(em_mean), label='Euler-Maruyama',color='white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('/scratch/xx84/girsanov/fk/bs/figure/dim_bs_loss_cnn.png')
plt.clf()


with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_cnn_time_mean.npy', 'rb') as f:
    cnn_mean = np.load(f)
with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_cnn_time_min.npy', 'rb') as f:
    cnn_min = np.load(f)
with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_cnn_time_max.npy', 'rb') as f:
    cnn_max = np.load(f)
with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_don_time_mean.npy', 'rb') as f:
    don_mean = np.load(f)
with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_don_time_min.npy', 'rb') as f:
    don_min = np.load(f)
with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_don_time_max.npy', 'rb') as f:
    don_max = np.load(f)
with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_em_time_mean.npy', 'rb') as f:
    em_mean = np.load(f)
with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_em_time_min.npy', 'rb') as f:
    em_min = np.load(f)
with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_em_time_max.npy', 'rb') as f:
    em_max = np.load(f)
#with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_em_time_max.npy', 'rb') as f:
#7    gir_max = np.load(f)
#with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_gir_time_mean.npy', 'rb') as f:
#    gir_mean = np.load(f)
#with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_gir_time_min.npy', 'rb') as f:
#    gir_min = np.load(f)
#with open('/scratch/xx84/girsanov/fk/bs/result/dim_bs_gir_time_max.npy', 'rb') as f:
#    gir_max = np.load(f)

ep = torch.arange(len(em_mean)) * 0.025

#plt.plot(ep, np.array(gir_mean), label='Girsanov', color='palevioletred')
#plt.fill_between(ep, np.array(gir_min), np.array(gir_max), alpha=0.2, color='lightpink')
ax = plt.subplot(111)
ax.plot(ep, np.array(cnn_mean), label='NGO', color='mediumturquoise')
ax.fill_between(ep, np.array(cnn_min), np.array(cnn_max), alpha=0.2, color='mediumturquoise')
ax.plot(ep, np.array(don_mean), label='DeepONet',color='slateblue')
ax.fill_between(ep, np.array(don_min), np.array(don_max), alpha=0.2,color='slateblue')
ax.plot(ep, np.array(em_mean), label='Euler-Maruyama',color='tomato')
ax.fill_between(ep, np.array(em_min), np.array(em_max), alpha=0.2,color='tomato')
ax.grid()
ax.set_ylabel('Computation Time')
ax.set_xlabel('Terminal Time')
ax.legend()
ax.plot(ep, np.array(cnn_mean), label='NGO', color='white')
ax.plot(ep, np.array(don_mean), label='DeepONet',color='white')
ax.plot(ep, np.array(em_mean), label='Euler-Maruyama',color='white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('/scratch/xx84/girsanov/fk/bs/figure/dim_bs_time_cnn.png')
plt.clf()