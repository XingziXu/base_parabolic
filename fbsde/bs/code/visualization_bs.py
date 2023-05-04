import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib

matplotlib.rcParams.update({'font.size': 15})

with open('/scratch/xx84/girsanov/fbsde/bs/result/10_bs_cnn_loss_mean.npy', 'rb') as f:
    cnn_mean = np.load(f)
with open('/scratch/xx84/girsanov/fbsde/bs/result/10_bs_cnn_loss_min.npy', 'rb') as f:
    cnn_min = np.load(f)
with open('/scratch/xx84/girsanov/fbsde/bs/result/10_bs_cnn_loss_max.npy', 'rb') as f:
    cnn_max = np.load(f)
with open('/scratch/xx84/girsanov/fbsde/bs/result/10_bs_em_loss_mean.npy', 'rb') as f:
    em_mean = np.load(f)
with open('/scratch/xx84/girsanov/fbsde/bs/result/10_bs_em_loss_min.npy', 'rb') as f:
    em_min = np.load(f)
with open('/scratch/xx84/girsanov/fbsde/bs/result/10_bs_em_loss_max.npy', 'rb') as f:
    em_max = np.load(f)
with open('/scratch/xx84/girsanov/fbsde/bs/result/10_bs_don_loss_mean.npy', 'rb') as f:
    don_mean = np.load(f)
with open('/scratch/xx84/girsanov/fbsde/bs/result/10_bs_don_loss_min.npy', 'rb') as f:
    don_min = np.load(f)
with open('/scratch/xx84/girsanov/fbsde/bs/result/10_bs_don_loss_max.npy', 'rb') as f:
    don_max = np.load(f)

ep = torch.arange(len(em_mean)) * 0.025

#plt.plot(ep, np.array(em_mean), label='Girsanov', color='palevioletred')
#plt.fill_between(ep, np.array(em_min), np.array(em_max), alpha=0.2, color='lightpink')

plt.plot(ep, np.array(cnn_mean), label='NGO', color='darkcyan')
plt.fill_between(ep, np.array(cnn_min), np.array(cnn_max), alpha=0.2, color='mediumturquoise')
#plt.ylim(0, np.array(don_max).max()+0.2)
#plt.plot(ep, np.array(don_mean), label='DeepONet',color='darkslateblue')
#plt.fill_between(ep, np.array(don_min), np.array(don_max), alpha=0.2,color='slateblue')
plt.plot(ep, np.array(em_mean), label='Euler-Maruyama',color='darkorange')
plt.fill_between(ep, np.array(em_min), np.array(em_max), alpha=0.2,color='bisque')
plt.ylabel('Normalized Error')
plt.xlabel('Terminal Time')
plt.legend()
plt.savefig('/scratch/xx84/girsanov/fbsde/bs/figure/10_bs_loss_cnn.png')
plt.clf()


with open('/scratch/xx84/girsanov/fbsde/bs/result/10_bs_cnn_time_mean.npy', 'rb') as f:
    cnn_mean = np.load(f)
with open('/scratch/xx84/girsanov/fbsde/bs/result/10_bs_cnn_time_min.npy', 'rb') as f:
    cnn_min = np.load(f)
with open('/scratch/xx84/girsanov/fbsde/bs/result/10_bs_cnn_time_max.npy', 'rb') as f:
    cnn_max = np.load(f)
with open('/scratch/xx84/girsanov/fbsde/bs/result/10_bs_don_time_mean.npy', 'rb') as f:
    don_mean = np.load(f)
with open('/scratch/xx84/girsanov/fbsde/bs/result/10_bs_don_time_min.npy', 'rb') as f:
    don_min = np.load(f)
with open('/scratch/xx84/girsanov/fbsde/bs/result/10_bs_don_time_max.npy', 'rb') as f:
    don_max = np.load(f)
with open('/scratch/xx84/girsanov/fbsde/bs/result/10_bs_em_time_mean.npy', 'rb') as f:
    em_mean = np.load(f)
with open('/scratch/xx84/girsanov/fbsde/bs/result/10_bs_em_time_min.npy', 'rb') as f:
    em_min = np.load(f)
with open('/scratch/xx84/girsanov/fbsde/bs/result/10_bs_em_time_max.npy', 'rb') as f:
    em_max = np.load(f)

ep = torch.arange(len(em_mean)) * 0.025

plt.plot(ep, np.array(cnn_mean)*1e3, label='NGO', color='darkcyan')
plt.fill_between(ep, np.array(cnn_min)*1e3, np.array(cnn_max)*1e3, alpha=0.2, color='mediumturquoise')
plt.plot(ep, np.array(don_mean)*1e3, label='DeepONet',color='darkslateblue')
plt.fill_between(ep, np.array(don_min)*1e3, np.array(don_max)*1e3, alpha=0.2,color='slateblue')
plt.plot(ep, np.array(em_mean)*1e3, label='Euler-Maruyama',color='darkorange')
plt.fill_between(ep, np.array(em_min)*1e3, np.array(em_max)*1e3, alpha=0.2,color='bisque')
plt.ylim(0, 0.15*1e3)
plt.ylabel('Computation Time')
plt.xlabel('Terminal Time')
plt.legend()
plt.savefig('/scratch/xx84/girsanov/fbsde/bs/figure/10_bs_time_cnn.png')
plt.clf()