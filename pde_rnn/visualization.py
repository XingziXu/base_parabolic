import matplotlib.pyplot as plt
import numpy as np
import torch

with open('/scratch/xx84/girsanov/pde_rnn/cnn_loss_mean.npy', 'rb') as f:
    cnn_mean = np.load(f)
with open('/scratch/xx84/girsanov/pde_rnn/cnn_loss_min.npy', 'rb') as f:
    cnn_min = np.load(f)
with open('/scratch/xx84/girsanov/pde_rnn/cnn_loss_max.npy', 'rb') as f:
    cnn_max = np.load(f)
with open('/scratch/xx84/girsanov/pde_rnn/gir_loss_mean.npy', 'rb') as f:
    gir_mean = np.load(f)
with open('/scratch/xx84/girsanov/pde_rnn/gir_loss_min.npy', 'rb') as f:
    gir_min = np.load(f)
with open('/scratch/xx84/girsanov/pde_rnn/gir_loss_max.npy', 'rb') as f:
    gir_max = np.load(f)
with open('/scratch/xx84/girsanov/pde_rnn/don_loss_mean.npy', 'rb') as f:
    don_mean = np.load(f)
with open('/scratch/xx84/girsanov/pde_rnn/don_loss_min.npy', 'rb') as f:
    don_min = np.load(f)
with open('/scratch/xx84/girsanov/pde_rnn/don_loss_max.npy', 'rb') as f:
    don_max = np.load(f)

ep = torch.arange(20) * 25

plt.plot(ep, np.array(gir_mean), label='Direct Girsanov', color='royalblue')
plt.fill_between(ep, np.array(gir_min), np.array(gir_max), alpha=0.2, color='cornflowerblue')
plt.plot(ep, np.array(cnn_mean), label='CNN', color='palevioletred')
plt.fill_between(ep, np.array(cnn_min), np.array(cnn_max), alpha=0.2, color='lightpink')
plt.ylim(0, np.array(don_max).max()+1)
plt.plot(ep, np.array(don_mean), label='DeepONet',color='darkslateblue')
plt.fill_between(ep, np.array(don_min), np.array(don_max), alpha=0.2,color='slateblue')
plt.plot(ep, ep * 0., label='Euler-Maruyama',color='darkorange')
plt.fill_between(ep, ep * 0., ep * 0., alpha=0.2,color='bisque')
plt.ylabel('Loss')
plt.xlabel('Terminal Time')
plt.legend()
plt.savefig('/scratch/xx84/girsanov/pde_rnn/loss_cnn.png')
plt.clf()


with open('/scratch/xx84/girsanov/pde_rnn/cnn_time_mean.npy', 'rb') as f:
    cnn_mean = np.load(f)
with open('/scratch/xx84/girsanov/pde_rnn/cnn_time_min.npy', 'rb') as f:
    cnn_min = np.load(f)
with open('/scratch/xx84/girsanov/pde_rnn/cnn_time_max.npy', 'rb') as f:
    cnn_max = np.load(f)
with open('/scratch/xx84/girsanov/pde_rnn/gir_time_mean.npy', 'rb') as f:
    gir_mean = np.load(f)
with open('/scratch/xx84/girsanov/pde_rnn/gir_time_min.npy', 'rb') as f:
    gir_min = np.load(f)
with open('/scratch/xx84/girsanov/pde_rnn/gir_time_max.npy', 'rb') as f:
    gir_max = np.load(f)
with open('/scratch/xx84/girsanov/pde_rnn/don_time_mean.npy', 'rb') as f:
    don_mean = np.load(f)
with open('/scratch/xx84/girsanov/pde_rnn/don_time_min.npy', 'rb') as f:
    don_min = np.load(f)
with open('/scratch/xx84/girsanov/pde_rnn/don_time_max.npy', 'rb') as f:
    don_max = np.load(f)
with open('/scratch/xx84/girsanov/pde_rnn/em_time_mean.npy', 'rb') as f:
    em_mean = np.load(f)
with open('/scratch/xx84/girsanov/pde_rnn/em_time_min.npy', 'rb') as f:
    em_min = np.load(f)
with open('/scratch/xx84/girsanov/pde_rnn/em_time_max.npy', 'rb') as f:
    em_max = np.load(f)

ep = torch.arange(20) * 0.1

plt.plot(ep, np.array(gir_mean), label='Direct Girsanov', color='royalblue')
plt.fill_between(ep, np.array(gir_min), np.array(gir_max), alpha=0.2, color='cornflowerblue')
plt.plot(ep, np.array(cnn_mean), label='CNN', color='palevioletred')
plt.fill_between(ep, np.array(cnn_min), np.array(cnn_max), alpha=0.2, color='lightpink')
plt.plot(ep, np.array(don_mean), label='DeepONet',color='darkslateblue')
plt.fill_between(ep, np.array(don_min), np.array(don_max), alpha=0.2,color='slateblue')
plt.plot(ep, np.array(em_mean), label='Euler-Maruyama',color='darkorange')
plt.fill_between(ep, np.array(em_min), np.array(em_max), alpha=0.2,color='bisque')
plt.ylabel('Loss')
plt.xlabel('Terminal Time')
plt.legend()
plt.savefig('/scratch/xx84/girsanov/pde_rnn/time_cnn.png')
plt.clf()