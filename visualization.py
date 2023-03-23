import matplotlib.pyplot as plt
import torch
import numpy as np

# loss_std = np.load('/scratch/xx84/girsanov/generative_modeling/2dgaussian_loss_std_0.npy')

# loss_meta = np.load('/scratch/xx84/girsanov/generative_modeling/2dgaussian_loss_nf_0.npy')

# bpd_std = np.load('/scratch/xx84/girsanov/generative_modeling/2dgaussian_bpd_std_0.npy')

# bpd_meta = np.load('/scratch/xx84/girsanov/generative_modeling/2dgaussian_bpd_nf_0.npy')

# plt.scatter(loss_std, bpd_std, c='r', label='std')
# plt.scatter(loss_meta, bpd_meta, c='b', label='meta')
# plt.xlabel('Wasserstein Distance')
# plt.ylabel('ELBO Loss')
# plt.legend()
# plt.savefig('/scratch/xx84/girsanov/visualized.png')


T = 0.01 * np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
gir = np.array([1.4668, 2.5989, 3.9424, 5.4984, 7.1330,9.0901, 11.2910, 13.3305, 15.3672, 17.4199, 19.3147, 21.2330, 23.3633, 25.6937, 27.6717])
rnn = np.array([2.4131, 3.6307, 4.6452, 5.9453, 7.4848, 9.2014, 11.1590, 13.0930, 14.7676, 16.8181, 18.6200, 20.5128, 22.6317, 24.7597, 26.6570])

plt.plot(T, gir, c='r', label='gir')
plt.plot(T, rnn, c='b', label='rnn')
plt.xlabel('Time')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/scratch/xx84/girsanov/visualized.png')