import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib

matplotlib.rcParams.update({'font.size': 15})

dim = 10
name = 'hjb'
pde = 'hjb'

with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_loss_mean.npy', 'rb') as f:
    cnn_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_loss_min.npy', 'rb') as f:
    cnn_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_loss_max.npy', 'rb') as f:
    cnn_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_loss_var.npy', 'rb') as f:
    cnn_var = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_em_loss_mean.npy', 'rb') as f:
    em_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_em_loss_min.npy', 'rb') as f:
    em_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_em_loss_max.npy', 'rb') as f:
    em_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_em_loss_var.npy', 'rb') as f:
    em_var = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_don_loss_mean.npy', 'rb') as f:
    don_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_don_loss_min.npy', 'rb') as f:
    don_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_don_loss_max.npy', 'rb') as f:
    don_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_don_loss_var.npy', 'rb') as f:
    don_var = torch.Tensor(np.load(f))


    #plt.plot(ep, np.array(em_mean), label='Girsanov', color='palevioletred')
    #plt.fill_between(ep, np.array(em_min), np.array(em_max), alpha=0.2, color='lightpink')

    


with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_time_mean.npy', 'rb') as f:
    cnn_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_time_min.npy', 'rb') as f:
    cnn_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_time_max.npy', 'rb') as f:
    cnn_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_time_var.npy', 'rb') as f:
    cnn_var_t = torch.Tensor(np.load(f))
#with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_gir_time_mean.npy', 'rb') as f:
#    gir_mean_t = torch.Tensor(np.load(f))
#with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_gir_time_min.npy', 'rb') as f:
#    gir_min_t = torch.Tensor(np.load(f))
#with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_gir_time_max.npy', 'rb') as f:
#    gir_max_t = torch.Tensor(np.load(f))
#with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_gir_time_var.npy', 'rb') as f:
#    gir_var_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_don_time_mean.npy', 'rb') as f:
    don_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_don_time_min.npy', 'rb') as f:
    don_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_don_time_max.npy', 'rb') as f:
    don_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_don_time_var.npy', 'rb') as f:
    don_var_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_em_time_mean.npy', 'rb') as f:
    em_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_em_time_min.npy', 'rb') as f:
    em_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_em_time_max.npy', 'rb') as f:
    em_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_em_time_var.npy', 'rb') as f:
    em_var_t = torch.Tensor(np.load(f))

ep = (torch.arange(len(cnn_mean))+1) * 0.025

ax = plt.subplot(221)
ax.set_yscale('log')
ax.plot(ep, np.array(cnn_mean), label='NGO', color='teal')
ax.fill_between(ep, np.array(cnn_mean-cnn_var), np.array(cnn_mean+cnn_var), alpha=0.2, color='teal')
#ax.set_ylim(0, np.array(don_max_t).max()+0.2)
ax.plot(ep, np.array(don_mean), label='DeepONet',color='slateblue')
ax.fill_between(ep, np.array(don_mean-don_var), np.array(don_mean+don_var), alpha=0.2,color='slateblue')
ax.plot(ep, np.array(em_mean), label='Euler-Maruyama',color='tomato')
ax.fill_between(ep, np.array(em_mean-em_var), np.array(em_mean+em_var), alpha=0.2,color='tomato')
#ax.plot(ep, np.array(gir_mean_t), label='Girsanov',color='darkorange')
#ax.fill_between(ep, np.array(gir_mean_t-gir_var_t), np.array(gir_mean_t+gir_var_t), alpha=0.2,color='darkorange')
ax.grid()
ax.set_ylabel('Percentage Error', labelpad=-2)
#ax.set_xlabel('Terminal Time', labelpad=0)
#ax.legend()
#ax.plot(ep, np.array(cnn_mean_t), color='white')
#ax.plot(ep, np.array(don_mean_t), color='white')
#ax.plot(ep, np.array(em_mean_t), color='white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


########################################################################################################
dim = 10
name = 'bsb'
pde = 'bsb'

with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_loss_mean.npy', 'rb') as f:
    cnn_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_loss_min.npy', 'rb') as f:
    cnn_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_loss_max.npy', 'rb') as f:
    cnn_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_loss_var.npy', 'rb') as f:
    cnn_var = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_em_loss_mean.npy', 'rb') as f:
    em_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_em_loss_min.npy', 'rb') as f:
    em_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_em_loss_max.npy', 'rb') as f:
    em_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_em_loss_var.npy', 'rb') as f:
    em_var = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_don_loss_mean.npy', 'rb') as f:
    don_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_don_loss_min.npy', 'rb') as f:
    don_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_don_loss_max.npy', 'rb') as f:
    don_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_don_loss_var.npy', 'rb') as f:
    don_var = torch.Tensor(np.load(f))


    #plt.plot(ep, np.array(em_mean), label='Girsanov', color='palevioletred')
    #plt.fill_between(ep, np.array(em_min), np.array(em_max), alpha=0.2, color='lightpink')

    


with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_time_mean.npy', 'rb') as f:
    cnn_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_time_min.npy', 'rb') as f:
    cnn_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_time_max.npy', 'rb') as f:
    cnn_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_time_var.npy', 'rb') as f:
    cnn_var_t = torch.Tensor(np.load(f))
#with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_gir_time_mean.npy', 'rb') as f:
#    gir_mean_t = torch.Tensor(np.load(f))
#with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_gir_time_min.npy', 'rb') as f:
#    gir_min_t = torch.Tensor(np.load(f))
#with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_gir_time_max.npy', 'rb') as f:
#    gir_max_t = torch.Tensor(np.load(f))
#with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_gir_time_var.npy', 'rb') as f:
#    gir_var_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_don_time_mean.npy', 'rb') as f:
    don_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_don_time_min.npy', 'rb') as f:
    don_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_don_time_max.npy', 'rb') as f:
    don_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_don_time_var.npy', 'rb') as f:
    don_var_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_em_time_mean.npy', 'rb') as f:
    em_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_em_time_min.npy', 'rb') as f:
    em_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_em_time_max.npy', 'rb') as f:
    em_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_em_time_var.npy', 'rb') as f:
    em_var_t = torch.Tensor(np.load(f))

ep = (torch.arange(len(cnn_mean))+1) * 0.025

ax = plt.subplot(222)
ax.set_yscale('log')
ax.plot(ep, np.array(cnn_mean), label='NGO', color='teal')
ax.fill_between(ep, np.array(cnn_mean-cnn_var), np.array(cnn_mean+cnn_var), alpha=0.2, color='teal')
#ax.set_ylim(0, np.array(don_max_t).max()+0.2)
ax.plot(ep, np.array(don_mean), label='DeepONet',color='slateblue')
ax.fill_between(ep, np.array(don_mean-don_var), np.array(don_mean+don_var), alpha=0.2,color='slateblue')
ax.plot(ep, np.array(em_mean), label='Euler-Maruyama',color='tomato')
ax.fill_between(ep, np.array(em_mean-em_var), np.array(em_mean+em_var), alpha=0.2,color='tomato')
#ax.plot(ep, np.array(gir_mean_t), label='Girsanov',color='darkorange')
#ax.fill_between(ep, np.array(gir_mean_t-gir_var_t), np.array(gir_mean_t+gir_var_t), alpha=0.2,color='darkorange')
ax.grid()
#ax.set_ylabel('Percentage Error', labelpad=-2)
#ax.set_xlabel('Terminal Time', labelpad=0)
#ax.legend()
#ax.plot(ep, np.array(cnn_mean_t), color='white')
#ax.plot(ep, np.array(don_mean_t), color='white')
#ax.plot(ep, np.array(em_mean_t), color='white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

########################################################################################################
dim = 10
name = 'bs'
pde = 'bs'

with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_loss_mean.npy', 'rb') as f:
    cnn_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_loss_min.npy', 'rb') as f:
    cnn_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_loss_max.npy', 'rb') as f:
    cnn_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_loss_var.npy', 'rb') as f:
    cnn_var = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_em_loss_mean.npy', 'rb') as f:
    em_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_em_loss_min.npy', 'rb') as f:
    em_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_em_loss_max.npy', 'rb') as f:
    em_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_em_loss_var.npy', 'rb') as f:
    em_var = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_don_loss_mean.npy', 'rb') as f:
    don_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_don_loss_min.npy', 'rb') as f:
    don_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_don_loss_max.npy', 'rb') as f:
    don_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_don_loss_var.npy', 'rb') as f:
    don_var = torch.Tensor(np.load(f))


    #plt.plot(ep, np.array(em_mean), label='Girsanov', color='palevioletred')
    #plt.fill_between(ep, np.array(em_min), np.array(em_max), alpha=0.2, color='lightpink')

    


with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_time_mean.npy', 'rb') as f:
    cnn_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_time_min.npy', 'rb') as f:
    cnn_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_time_max.npy', 'rb') as f:
    cnn_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_time_var.npy', 'rb') as f:
    cnn_var_t = torch.Tensor(np.load(f))
#with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_gir_time_mean.npy', 'rb') as f:
#    gir_mean_t = torch.Tensor(np.load(f))
#with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_gir_time_min.npy', 'rb') as f:
#    gir_min_t = torch.Tensor(np.load(f))
#with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_gir_time_max.npy', 'rb') as f:
#    gir_max_t = torch.Tensor(np.load(f))
#with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_gir_time_var.npy', 'rb') as f:
#    gir_var_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_don_time_mean.npy', 'rb') as f:
    don_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_don_time_min.npy', 'rb') as f:
    don_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_don_time_max.npy', 'rb') as f:
    don_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_don_time_var.npy', 'rb') as f:
    don_var_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_em_time_mean.npy', 'rb') as f:
    em_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_em_time_min.npy', 'rb') as f:
    em_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_em_time_max.npy', 'rb') as f:
    em_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_em_time_var.npy', 'rb') as f:
    em_var_t = torch.Tensor(np.load(f))

ep = (torch.arange(len(cnn_mean))+1) * 0.025

ax = plt.subplot(223)
ax.set_yscale('log')
ax.plot(ep, np.array(cnn_mean), label='NGO', color='teal')
ax.fill_between(ep, np.array(cnn_mean-cnn_var), np.array(cnn_mean+cnn_var), alpha=0.2, color='teal')
#ax.set_ylim(0, np.array(don_max_t).max()+0.2)
ax.plot(ep, np.array(don_mean), label='DeepONet',color='slateblue')
ax.fill_between(ep, np.array(don_mean-don_var), np.array(don_mean+don_var), alpha=0.2,color='slateblue')
ax.plot(ep, np.array(em_mean), label='Euler-Maruyama',color='tomato')
ax.fill_between(ep, np.array(em_mean-em_var), np.array(em_mean+em_var), alpha=0.2,color='tomato')
#ax.plot(ep, np.array(gir_mean_t), label='Girsanov',color='darkorange')
#ax.fill_between(ep, np.array(gir_mean_t-gir_var_t), np.array(gir_mean_t+gir_var_t), alpha=0.2,color='darkorange')
ax.grid()
ax.set_ylabel('Percentage Error', labelpad=-2)
ax.set_xlabel('Terminal Time', labelpad=0)
#ax.legend()
#ax.plot(ep, np.array(cnn_mean_t), color='white')
#ax.plot(ep, np.array(don_mean_t), color='white')
#ax.plot(ep, np.array(em_mean_t), color='white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

########################################################################################################
dim = 10
name = 'ou'
pde = 'ou'

with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_loss_mean.npy', 'rb') as f:
    cnn_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_loss_min.npy', 'rb') as f:
    cnn_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_loss_max.npy', 'rb') as f:
    cnn_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_loss_var.npy', 'rb') as f:
    cnn_var = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_em_loss_mean.npy', 'rb') as f:
    em_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_em_loss_min.npy', 'rb') as f:
    em_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_em_loss_max.npy', 'rb') as f:
    em_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_em_loss_var.npy', 'rb') as f:
    em_var = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_don_loss_mean.npy', 'rb') as f:
    don_mean = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_don_loss_min.npy', 'rb') as f:
    don_min = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_don_loss_max.npy', 'rb') as f:
    don_max = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_don_loss_var.npy', 'rb') as f:
    don_var = torch.Tensor(np.load(f))


    #plt.plot(ep, np.array(em_mean), label='Girsanov', color='palevioletred')
    #plt.fill_between(ep, np.array(em_min), np.array(em_max), alpha=0.2, color='lightpink')

    


with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_time_mean.npy', 'rb') as f:
    cnn_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_time_min.npy', 'rb') as f:
    cnn_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_time_max.npy', 'rb') as f:
    cnn_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_cnn_time_var.npy', 'rb') as f:
    cnn_var_t = torch.Tensor(np.load(f))
#with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_gir_time_mean.npy', 'rb') as f:
#    gir_mean_t = torch.Tensor(np.load(f))
#with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_gir_time_min.npy', 'rb') as f:
#    gir_min_t = torch.Tensor(np.load(f))
#with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_gir_time_max.npy', 'rb') as f:
#    gir_max_t = torch.Tensor(np.load(f))
#with open('/scratch/xx84/girsanov/fbsde/'+name+'/result/'+str(dim)+'_'+pde+'_gir_time_var.npy', 'rb') as f:
#    gir_var_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_don_time_mean.npy', 'rb') as f:
    don_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_don_time_min.npy', 'rb') as f:
    don_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_don_time_max.npy', 'rb') as f:
    don_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_don_time_var.npy', 'rb') as f:
    don_var_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_em_time_mean.npy', 'rb') as f:
    em_mean_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_em_time_min.npy', 'rb') as f:
    em_min_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_em_time_max.npy', 'rb') as f:
    em_max_t = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/'+name+'/result/'+str(dim)+'_'+pde+'_em_time_var.npy', 'rb') as f:
    em_var_t = torch.Tensor(np.load(f))

ep = (torch.arange(len(cnn_mean))+1) * 0.025

ax = plt.subplot(224)
ax.set_yscale('log')
ax.plot(ep, np.array(cnn_mean), label='NGO', color='teal')
ax.fill_between(ep, np.array(cnn_mean-cnn_var), np.array(cnn_mean+cnn_var), alpha=0.2, color='teal')
#ax.set_ylim(0, np.array(don_max_t).max()+0.2)
ax.plot(ep, np.array(don_mean), label='DeepONet',color='slateblue')
ax.fill_between(ep, np.array(don_mean-don_var), np.array(don_mean+don_var), alpha=0.2,color='slateblue')
ax.plot(ep, np.array(em_mean), label='Euler-Maruyama',color='tomato')
ax.fill_between(ep, np.array(em_mean-em_var), np.array(em_mean+em_var), alpha=0.2,color='tomato')
#ax.plot(ep, np.array(gir_mean_t), label='Girsanov',color='darkorange')
#ax.fill_between(ep, np.array(gir_mean_t-gir_var_t), np.array(gir_mean_t+gir_var_t), alpha=0.2,color='darkorange')
ax.grid()
#ax.set_ylabel('Percentage Error', labelpad=-2)
ax.set_xlabel('Terminal Time', labelpad=0)
#ax.legend()
#ax.plot(ep, np.array(cnn_mean_t), color='white')
#ax.plot(ep, np.array(don_mean_t), color='white')
#ax.plot(ep, np.array(em_mean_t), color='white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/scratch/xx84/girsanov/fbsde/hjb/figure/'+str(dim)+'_loss_exact.png')
plt.clf()