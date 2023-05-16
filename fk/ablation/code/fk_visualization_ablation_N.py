import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib

matplotlib.rcParams.update({'font.size': 45})

with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_cnn_loss_mean.npy', 'rb') as f:
    cnn_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_cnn_loss_min.npy', 'rb') as f:
    cnn_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_cnn_loss_max.npy', 'rb') as f:
    cnn_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_cnn_loss_var.npy', 'rb') as f:
    cnn_var = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_gir_loss_mean.npy', 'rb') as f:
    gir_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_gir_loss_min.npy', 'rb') as f:
    gir_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_gir_loss_max.npy', 'rb') as f:
    gir_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_gir_loss_var.npy', 'rb') as f:
    gir_var = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_don_loss_mean.npy', 'rb') as f:
    don_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_don_loss_min.npy', 'rb') as f:
    don_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_don_loss_max.npy', 'rb') as f:
    don_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_don_loss_var.npy', 'rb') as f:
    don_var = torch.Tensor(np.load(f))

length_data = len(cnn_mean)

    #plt.plot(ep, np.array(em_mean), label='Girsanov', color='palevioletred')
    #plt.fill_between(ep, np.array(em_min), np.array(em_max), alpha=0.2, color='lightpink')

    


with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_cnn_time_mean.npy', 'rb') as f:
    cnn_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_cnn_time_min.npy', 'rb') as f:
    cnn_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_cnn_time_max.npy', 'rb') as f:
    cnn_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_cnn_time_var.npy', 'rb') as f:
    cnn_var_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_gir_time_mean.npy', 'rb') as f:
    gir_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_gir_time_min.npy', 'rb') as f:
    gir_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_gir_time_max.npy', 'rb') as f:
    gir_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_gir_time_var.npy', 'rb') as f:
    gir_var_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_don_time_mean.npy', 'rb') as f:
    don_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_don_time_min.npy', 'rb') as f:
    don_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_don_time_max.npy', 'rb') as f:
    don_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_don_time_var.npy', 'rb') as f:
    don_var_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_em_time_mean.npy', 'rb') as f:
    em_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_em_time_min.npy', 'rb') as f:
    em_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_em_time_max.npy', 'rb') as f:
    em_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_em_time_var.npy', 'rb') as f:
    em_var_t = torch.Tensor(np.load(f))

fig, ax = plt.subplots(2,3)
fig.set_size_inches(30, 20)
#ax = plt.subplot(224)
ax[0,0].set_yscale('log')

ep = (torch.arange(len(gir_mean))+1) * 50

ax[0,0].set_yscale('log')
ax[0,0].plot(ep, np.array(gir_mean), label='GIR', color='darkorange', lw=3)
ax[0,0].fill_between(ep, np.array(gir_mean-gir_var), np.array(gir_mean+gir_var), alpha=0.2, color='darkorange')
ax[0,0].plot(ep, np.array(cnn_mean), label='NGO', color='teal', lw=3)
ax[0,0].fill_between(ep, np.array(cnn_mean-cnn_var), np.array(cnn_mean+cnn_var), alpha=0.2, color='teal')
ax[0,0].plot(ep, np.array(don_mean), label='DON',color='slateblue', lw=3)
ax[0,0].fill_between(ep, np.array(don_mean-don_var), np.array(don_mean+don_var), alpha=0.2,color='slateblue')
#plt.ylim(0, 0.02*1e3)
ax[0,0].set_ylabel('Normalized Error')
#ax[0,0].set_xlabel('Number of Samples')
ax[0,0].grid()

ax[1,0].set_yscale('log')
ax[1,0].plot(ep, np.array(cnn_mean_t), label='NGO', color='teal', lw=3)
ax[1,0].fill_between(ep, np.array(cnn_mean_t-cnn_var_t), np.array(cnn_mean_t+cnn_var_t), alpha=0.2, color='teal')
ax[1,0].plot(ep, np.array(gir_mean_t), label='GIR', color='darkorange', lw=3)
ax[1,0].fill_between(ep, np.array(gir_mean_t-gir_var_t), np.array(gir_mean_t+gir_var_t), alpha=0.2, color='darkorange')
ax[1,0].plot(ep, np.array(em_mean_t), label='E-M', color='tomato', lw=3)
ax[1,0].fill_between(ep, np.array(em_mean_t-em_var_t), np.array(em_mean_t+em_var_t), alpha=0.2, color='tomato')
ax[1,0].plot(ep, np.array(don_mean_t), label='DON',color='slateblue', lw=3)
ax[1,0].fill_between(ep, np.array(don_mean_t-don_var_t), np.array(don_mean_t+don_var_t), alpha=0.2,color='slateblue')
ax[1,0].set_ylabel('Inference Time')
ax[1,0].set_xlabel('Number of Samples')
ax[1,0].grid()


##############################################################################################

with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_cnn_loss_mean.npy', 'rb') as f:
    ou_cnn_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_cnn_loss_min.npy', 'rb') as f:
    ou_cnn_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_cnn_loss_max.npy', 'rb') as f:
    ou_cnn_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_cnn_loss_var.npy', 'rb') as f:
    ou_cnn_var = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_gir_loss_mean.npy', 'rb') as f:
    ou_gir_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_gir_loss_min.npy', 'rb') as f:
    ou_gir_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_gir_loss_max.npy', 'rb') as f:
    ou_gir_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_gir_loss_var.npy', 'rb') as f:
    ou_gir_var = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_don_loss_mean.npy', 'rb') as f:
    ou_don_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_don_loss_min.npy', 'rb') as f:
    ou_don_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_don_loss_max.npy', 'rb') as f:
    ou_don_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_don_loss_var.npy', 'rb') as f:
    ou_don_var = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_em_loss_mean.npy', 'rb') as f:
    ou_em_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_em_loss_min.npy', 'rb') as f:
    ou_em_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_em_loss_max.npy', 'rb') as f:
    ou_em_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_em_loss_var.npy', 'rb') as f:
    ou_em_var = torch.Tensor(np.load(f))

length_data = len(ou_cnn_mean)

    #plt.plot(ep, np.array(em_mean), label='Girsanov', color='palevioletred')
    #plt.fill_between(ep, np.array(em_min), np.array(em_max), alpha=0.2, color='lightpink')

    


with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_cnn_time_mean.npy', 'rb') as f:
    ou_cnn_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_cnn_time_min.npy', 'rb') as f:
    ou_cnn_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_cnn_time_max.npy', 'rb') as f:
    ou_cnn_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_cnn_time_var.npy', 'rb') as f:
    ou_cnn_var_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_gir_time_mean.npy', 'rb') as f:
    ou_gir_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_gir_time_min.npy', 'rb') as f:
    ou_gir_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_gir_time_max.npy', 'rb') as f:
    ou_gir_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_gir_time_var.npy', 'rb') as f:
    ou_gir_var_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_don_time_mean.npy', 'rb') as f:
    ou_don_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_don_time_min.npy', 'rb') as f:
    ou_don_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_don_time_max.npy', 'rb') as f:
    ou_don_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_don_time_var.npy', 'rb') as f:
    ou_don_var_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_em_time_mean.npy', 'rb') as f:
    ou_em_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_em_time_min.npy', 'rb') as f:
    ou_em_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_em_time_max.npy', 'rb') as f:
    ou_em_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_ou_em_time_var.npy', 'rb') as f:
    ou_em_var_t = torch.Tensor(np.load(f))

ax[0,1].set_yscale('log')
ax[0,1].plot(ep, np.array(ou_gir_mean), label='GIR', color='darkorange', lw=3)
ax[0,1].fill_between(ep, np.array(ou_gir_mean-ou_gir_var), np.array(ou_gir_mean+ou_gir_var), alpha=0.2, color='darkorange')
ax[0,1].plot(ep, np.array(ou_em_mean), label='E-M', color='tomato', lw=3)
ax[0,1].fill_between(ep, np.array(ou_em_mean-ou_em_var), np.array(ou_em_mean+ou_em_var), alpha=0.2, color='tomato')
ax[0,1].plot(ep, np.array(ou_cnn_mean), label='NGO', color='teal', lw=3)
ax[0,1].fill_between(ep, np.array(ou_cnn_mean-ou_cnn_var), np.array(ou_cnn_mean+ou_cnn_var), alpha=0.2, color='teal')
ax[0,1].plot(ep, np.array(ou_don_mean), label='DON',color='slateblue', lw=3)
ax[0,1].fill_between(ep, np.array(ou_don_mean-ou_don_var), np.array(ou_don_mean+ou_don_var), alpha=0.2,color='slateblue')
#plt.ylim(0, 0.02*1e3)
#ax[0,1].set_ylabel('Normalized Error')
#ax[0,1].set_xlabel('Number of Samples')
ax[0,1].grid()

ax[1,1].set_yscale('log')
ax[1,1].plot(ep, np.array(ou_em_mean_t), label='E-M', color='tomato', lw=3)
ax[1,1].fill_between(ep, np.array(ou_em_mean_t-ou_em_var_t), np.array(ou_em_mean_t+ou_em_var_t), alpha=0.2, color='tomato')
ax[1,1].plot(ep, np.array(ou_cnn_mean_t), label='NGO', color='teal', lw=3)
ax[1,1].fill_between(ep, np.array(ou_cnn_mean_t-ou_cnn_var_t), np.array(ou_cnn_mean_t+ou_cnn_var_t), alpha=0.2, color='teal')
ax[1,1].plot(ep, np.array(ou_gir_mean_t), label='GIR', color='darkorange', lw=3)
ax[1,1].fill_between(ep, np.array(ou_gir_mean_t-ou_gir_var_t), np.array(ou_gir_mean_t+ou_gir_var_t), alpha=0.2, color='darkorange')
#ax[0,1].ylim(0, np.array(don_mean).max()+0.2)
#plt.ylim(0., 1.)
ax[1,1].plot(ep, np.array(ou_don_mean_t), label='DON',color='slateblue', lw=3)
ax[1,1].fill_between(ep, np.array(ou_don_mean_t-ou_don_var_t), np.array(ou_don_mean_t+ou_don_var_t), alpha=0.2,color='slateblue')
#ax[1,1].set_ylabel('Inference Time')
ax[1,1].set_xlabel('Number of Samples')
ax[1,1].grid()

##############################################################################################

with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_cnn_loss_mean.npy', 'rb') as f:
    bs_cnn_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_cnn_loss_min.npy', 'rb') as f:
    bs_cnn_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_cnn_loss_max.npy', 'rb') as f:
    bs_cnn_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_cnn_loss_var.npy', 'rb') as f:
    bs_cnn_var = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_don_loss_mean.npy', 'rb') as f:
    bs_don_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_don_loss_min.npy', 'rb') as f:
    bs_don_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_don_loss_max.npy', 'rb') as f:
    bs_don_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_don_loss_var.npy', 'rb') as f:
    bs_don_var = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_em_loss_mean.npy', 'rb') as f:
    bs_em_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_em_loss_min.npy', 'rb') as f:
    bs_em_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_em_loss_max.npy', 'rb') as f:
    bs_em_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_em_loss_var.npy', 'rb') as f:
    bs_em_var = torch.Tensor(np.load(f))

length_data = len(bs_cnn_mean)

    #plt.plot(ep, np.array(em_mean), label='Girsanov', color='palevioletred')
    #plt.fill_between(ep, np.array(em_min), np.array(em_max), alpha=0.2, color='lightpink')

with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_cnn_time_mean.npy', 'rb') as f:
    bs_cnn_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_cnn_time_min.npy', 'rb') as f:
    bs_cnn_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_cnn_time_max.npy', 'rb') as f:
    bs_cnn_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_cnn_time_var.npy', 'rb') as f:
    bs_cnn_var_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_don_time_mean.npy', 'rb') as f:
    bs_don_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_don_time_min.npy', 'rb') as f:
    bs_don_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_don_time_max.npy', 'rb') as f:
    bs_don_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_don_time_var.npy', 'rb') as f:
    bs_don_var_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_em_time_mean.npy', 'rb') as f:
    bs_em_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_em_time_min.npy', 'rb') as f:
    bs_em_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_em_time_max.npy', 'rb') as f:
    bs_em_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ablation/result_N/N_bs_em_time_var.npy', 'rb') as f:
    bs_em_var_t = torch.Tensor(np.load(f))

ax[0,2].set_yscale('log')
ax[0,2].plot(ep, np.array(bs_em_mean), label='E-M', color='tomato', lw=3)
ax[0,2].fill_between(ep, np.array(bs_em_mean-bs_em_var), np.array(bs_em_mean+bs_em_var), alpha=0.2, color='tomato')
ax[0,2].plot(ep, np.array(bs_cnn_mean), label='NGO', color='teal', lw=3)
ax[0,2].fill_between(ep, np.array(bs_cnn_mean-bs_cnn_var), np.array(bs_cnn_mean+bs_cnn_var), alpha=0.2, color='teal')
ax[0,2].plot(ep, np.array(bs_don_mean), label='DON',color='slateblue', lw=3)
ax[0,2].fill_between(ep, np.array(bs_don_mean-bs_don_var), np.array(bs_don_mean+bs_don_var), alpha=0.2,color='slateblue')
#plt.ylim(0, 0.02*1e3)
#ax[0,2].set_ylabel('Normalized Error')
#ax[0,2].set_xlabel('Number of Samples')
ax[0,2].grid()

ax[1,2].set_yscale('log')
ax[1,2].plot(ep, np.array(bs_cnn_mean_t), label='NGO', color='teal', lw=3)
ax[1,2].fill_between(ep, np.array(bs_cnn_mean_t-bs_cnn_var_t), np.array(bs_cnn_mean_t+bs_cnn_var_t), alpha=0.2, color='teal')
ax[1,2].plot(ep, np.array(bs_em_mean_t), label='E-M', color='tomato', lw=3)
ax[1,2].fill_between(ep, np.array(bs_em_mean_t-bs_em_var_t), np.array(bs_em_mean_t+bs_em_var_t), alpha=0.2, color='tomato')
#ax[0,1].ylim(0, np.array(don_mean).max()+0.2)
#plt.ylim(0., 1.)
ax[1,2].plot(ep, np.array(bs_don_mean_t), label='DON',color='slateblue', lw=3)
ax[1,2].fill_between(ep, np.array(bs_don_mean_t-bs_don_var_t), np.array(bs_don_mean_t+bs_don_var_t), alpha=0.2,color='slateblue')
#ax[1,2].set_ylabel('Inference Time')
ax[1,2].set_xlabel('Number of Samples')
ax[1,2].grid()

lines_labels = [ax[1,0].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels)
plt.tight_layout()
plt.savefig('/scratch/xx84/girsanov/fk/ablation/figure_N/N_ablation_fk.pdf')
plt.clf()