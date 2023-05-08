import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib

matplotlib.rcParams.update({'font.size': 15})

with open('/scratch/xx84/girsanov/fk/ablation/result/dim_cnn_loss_mean.npy', 'rb') as f:
    cnn_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_cnn_loss_min.npy', 'rb') as f:
    cnn_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_cnn_loss_max.npy', 'rb') as f:
    cnn_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_cnn_loss_var.npy', 'rb') as f:
    cnn_var = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_gir_loss_mean.npy', 'rb') as f:
    gir_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_gir_loss_min.npy', 'rb') as f:
    gir_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_gir_loss_max.npy', 'rb') as f:
    gir_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_gir_loss_var.npy', 'rb') as f:
    gir_var = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_don_loss_mean.npy', 'rb') as f:
    don_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_don_loss_min.npy', 'rb') as f:
    don_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_don_loss_max.npy', 'rb') as f:
    don_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_don_loss_var.npy', 'rb') as f:
    don_var = torch.Tensor(np.load(f))

length_data = len(cnn_mean)

    #plt.plot(ep, np.array(em_mean), label='Girsanov', color='palevioletred')
    #plt.fill_between(ep, np.array(em_min), np.array(em_max), alpha=0.2, color='lightpink')

    


with open('/scratch/xx84/girsanov/fk/ablation/result/dim_cnn_time_mean.npy', 'rb') as f:
    cnn_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_cnn_time_min.npy', 'rb') as f:
    cnn_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_cnn_time_max.npy', 'rb') as f:
    cnn_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_cnn_time_var.npy', 'rb') as f:
    cnn_var_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_gir_time_mean.npy', 'rb') as f:
    gir_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_gir_time_min.npy', 'rb') as f:
    gir_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_gir_time_max.npy', 'rb') as f:
    gir_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_gir_time_var.npy', 'rb') as f:
    gir_var_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_don_time_mean.npy', 'rb') as f:
    don_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_don_time_min.npy', 'rb') as f:
    don_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_don_time_max.npy', 'rb') as f:
    don_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_don_time_var.npy', 'rb') as f:
    don_var_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_em_time_mean.npy', 'rb') as f:
    em_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_em_time_min.npy', 'rb') as f:
    em_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_em_time_max.npy', 'rb') as f:
    em_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_em_time_var.npy', 'rb') as f:
    em_var_t = torch.Tensor(np.load(f))

ep = torch.arange(len(gir_mean))+1

plt.plot(ep, np.array(gir_mean_t)*1e3, label='Girsanov', color='palevioletred')
plt.fill_between(ep, np.array(gir_mean_t-gir_var_t)*1e3, np.array(gir_mean_t+gir_var_t)*1e3, alpha=0.2, color='lightpink')
plt.plot(ep, np.array(cnn_mean_t)*1e3, label='NGO', color='darkcyan')
plt.fill_between(ep, np.array(cnn_mean_t-cnn_var_t)*1e3, np.array(cnn_mean_t+cnn_var_t)*1e3, alpha=0.2, color='mediumturquoise')
plt.plot(ep, np.array(don_mean_t)*1e3, label='DeepONet',color='darkslateblue')
plt.fill_between(ep, np.array(don_mean_t-don_var_t)*1e3, np.array(don_mean_t+don_var_t)*1e3, alpha=0.2,color='slateblue')
plt.plot(ep, np.array(em_mean_t)*1e3, label='Euler-Maruyama',color='darkorange')
plt.fill_between(ep, np.array(em_mean_t-em_var_t)*1e3, np.array(em_mean_t+em_var_t)*1e3, alpha=0.2,color='bisque')
#plt.ylim(0, 0.02*1e3)
plt.ylabel('Computation Time')
plt.xlabel('Dimension')
plt.legend()
plt.savefig('/scratch/xx84/girsanov/fk/ablation/figure/dim_time_cnn.pdf')
plt.clf()

plt.plot(ep, np.array(cnn_mean), label='NGO', color='darkcyan')
plt.fill_between(ep, np.array(cnn_mean-cnn_var), np.array(cnn_mean+cnn_var), alpha=0.2, color='mediumturquoise')
plt.plot(ep, np.array(gir_mean), label='Girsanov', color='palevioletred')
plt.fill_between(ep, np.array(gir_mean-gir_var), np.array(gir_mean+gir_var), alpha=0.2, color='lightpink')
#plt.ylim(0, np.array(don_max).max()+0.2)
plt.ylim(0., 1.)
plt.plot(ep, np.array(don_mean), label='DeepONet',color='darkslateblue')
plt.fill_between(ep, np.array(don_mean-don_var), np.array(don_mean+don_var), alpha=0.2,color='slateblue')
plt.ylabel('Normalized Error')
plt.xlabel('Dimension')
plt.legend()
plt.savefig('/scratch/xx84/girsanov/fk/ablation/figure/dim_loss_cnn.pdf')
plt.clf()

#############################################################################################

with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_cnn_loss_mean.npy', 'rb') as f:
    cnn_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_cnn_loss_min.npy', 'rb') as f:
    cnn_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_cnn_loss_max.npy', 'rb') as f:
    cnn_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_cnn_loss_var.npy', 'rb') as f:
    cnn_var = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_gir_loss_mean.npy', 'rb') as f:
    gir_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_gir_loss_min.npy', 'rb') as f:
    gir_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_gir_loss_max.npy', 'rb') as f:
    gir_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_gir_loss_var.npy', 'rb') as f:
    gir_var = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_don_loss_mean.npy', 'rb') as f:
    don_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_don_loss_min.npy', 'rb') as f:
    don_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_don_loss_max.npy', 'rb') as f:
    don_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_don_loss_var.npy', 'rb') as f:
    don_var = torch.Tensor(np.load(f))

length_data = len(cnn_mean)

    #plt.plot(ep, np.array(em_mean), label='Girsanov', color='palevioletred')
    #plt.fill_between(ep, np.array(em_min), np.array(em_max), alpha=0.2, color='lightpink')

    


with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_cnn_time_mean.npy', 'rb') as f:
    cnn_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_cnn_time_min.npy', 'rb') as f:
    cnn_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_cnn_time_max.npy', 'rb') as f:
    cnn_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_cnn_time_var.npy', 'rb') as f:
    cnn_var_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_gir_time_mean.npy', 'rb') as f:
    gir_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_gir_time_min.npy', 'rb') as f:
    gir_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_gir_time_max.npy', 'rb') as f:
    gir_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_gir_time_var.npy', 'rb') as f:
    gir_var_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_don_time_mean.npy', 'rb') as f:
    don_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_don_time_min.npy', 'rb') as f:
    don_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_don_time_max.npy', 'rb') as f:
    don_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_don_time_var.npy', 'rb') as f:
    don_var_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_em_time_mean.npy', 'rb') as f:
    em_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_em_time_min.npy', 'rb') as f:
    em_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_em_time_max.npy', 'rb') as f:
    em_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result/dim_ou_em_time_var.npy', 'rb') as f:
    em_var_t = torch.Tensor(np.load(f))

ep = torch.arange(len(gir_mean))+1

plt.plot(ep, np.array(gir_mean_t)*1e3, label='Girsanov', color='palevioletred')
plt.fill_between(ep, np.array(gir_mean_t-gir_var_t)*1e3, np.array(gir_mean_t+gir_var_t)*1e3, alpha=0.2, color='lightpink')
plt.plot(ep, np.array(cnn_mean_t)*1e3, label='NGO', color='darkcyan')
plt.fill_between(ep, np.array(cnn_mean_t-cnn_var_t)*1e3, np.array(cnn_mean_t+cnn_var_t)*1e3, alpha=0.2, color='mediumturquoise')
plt.plot(ep, np.array(don_mean_t)*1e3, label='DeepONet',color='darkslateblue')
plt.fill_between(ep, np.array(don_mean_t-don_var_t)*1e3, np.array(don_mean_t+don_var_t)*1e3, alpha=0.2,color='slateblue')
plt.plot(ep, np.array(em_mean_t)*1e3, label='Euler-Maruyama',color='darkorange')
plt.fill_between(ep, np.array(em_mean_t-em_var_t)*1e3, np.array(em_mean_t+em_var_t)*1e3, alpha=0.2,color='bisque')
#plt.ylim(0, 0.02*1e3)
plt.ylabel('Computation Time')
plt.xlabel('Dimension')
plt.legend()
plt.savefig('/scratch/xx84/girsanov/fk/ablation/figure/dim_ou_time_cnn.pdf')
plt.clf()

plt.plot(ep, np.array(cnn_mean), label='NGO', color='darkcyan')
plt.fill_between(ep, np.array(cnn_mean-cnn_var), np.array(cnn_mean+cnn_var), alpha=0.2, color='mediumturquoise')
plt.plot(ep, np.array(gir_mean), label='Girsanov', color='palevioletred')
plt.fill_between(ep, np.array(gir_mean-gir_var), np.array(gir_mean+gir_var), alpha=0.2, color='lightpink')
#plt.ylim(0, np.array(don_max).max()+0.2)
plt.ylim(0., 1.)
plt.plot(ep, np.array(don_mean), label='DeepONet',color='darkslateblue')
plt.fill_between(ep, np.array(don_mean-don_var), np.array(don_mean+don_var), alpha=0.2,color='slateblue')
plt.ylabel('Normalized Error')
plt.xlabel('Dimension')
plt.legend()
plt.savefig('/scratch/xx84/girsanov/fk/ablation/figure/dim_ou_loss_cnn.pdf')
plt.clf()