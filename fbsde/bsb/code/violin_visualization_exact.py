import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib
import seaborn as sns
import pandas as pd

matplotlib.rcParams.update({'font.size': 15})

dim = 10

cnn = torch.zeros(41, 25)
don = torch.zeros(41, 25)
em = torch.zeros(41, 25)
for i in range(1,25):
    num_time = i * 10
    with open('/scratch/xx84/girsanov/fbsde/bsb/result/'+str(dim)+'_'+str(num_time)+'_bsb_cnn_loss.npy', 'rb') as f:
        cnn_current = torch.Tensor(np.load(f))
    with open('/scratch/xx84/girsanov/fbsde/bsb/result/'+str(dim)+'_'+str(num_time)+'_bsb_don_loss.npy', 'rb') as f:
        don_current = torch.Tensor(np.load(f))
    with open('/scratch/xx84/girsanov/fbsde/bsb/result/'+str(dim)+'_'+str(num_time)+'_bsb_em_loss.npy', 'rb') as f:
        em_current = torch.Tensor(np.load(f))
    cnn[:,i-1] = cnn_current[0,:]
    don[:,i-1] = don_current[0,:]
    em[:,i-1] = em_current[0,:]

cnn = torch.Tensor(cnn.numpy()[np.where(cnn.numpy()!=0)])
don = torch.Tensor(don.numpy()[np.where(don.numpy()!=0)])
em = torch.Tensor(em.numpy()[np.where(em.numpy()!=0)])

cnn_t = torch.zeros(41, 25)
don_t = torch.zeros(41, 25)
em_t = torch.zeros(41, 25)
for i in range(1,25):
    num_time = i * 10
    with open('/scratch/xx84/girsanov/fbsde/bsb/result/'+str(dim)+'_'+str(num_time)+'_bsb_cnn_time.npy', 'rb') as f:
        cnn_current_t = torch.Tensor(np.load(f))
    with open('/scratch/xx84/girsanov/fbsde/bsb/result/'+str(dim)+'_'+str(num_time)+'_bsb_don_time.npy', 'rb') as f:
        don_current_t = torch.Tensor(np.load(f))
    with open('/scratch/xx84/girsanov/fbsde/bsb/result/'+str(dim)+'_'+str(num_time)+'_bsb_em_time.npy', 'rb') as f:
        em_current_t = torch.Tensor(np.load(f))
    cnn_t[:,i-1] = cnn_current_t[0,:]
    don_t[:,i-1] = don_current_t[0,:]
    em_t[:,i-1] = em_current_t[0,:]

cnn = torch.Tensor(cnn.numpy()[np.where(cnn.numpy()!=0)])
don = torch.Tensor(don.numpy()[np.where(don.numpy()!=0)])
em = torch.Tensor(em.numpy()[np.where(em.numpy()!=0)])

cnn_t = torch.Tensor(cnn_t.numpy()[np.where(cnn_t.numpy()!=0)])
don_t = torch.Tensor(don_t.numpy()[np.where(don_t.numpy()!=0)])
em_t = torch.Tensor(em_t.numpy()[np.where(em_t.numpy()!=0)])

loss = torch.concatenate((cnn,don,em),dim=0)
time = torch.concatenate((cnn_t,don_t,em_t),dim=0)
all = torch.concatenate((loss,time),dim=0)

group_l = ['NGO'] * len(cnn) + ['DeepONet'] * len(don) + ['Euler-Maruyama'] * len(em) +['NGO'] * len(cnn) + ['DeepONet'] * len(don) + ['Euler-Maruyama'] * len(em)
group_hue = ['Normalized Error'] * (len(cnn)+len(don)+len(em)) + ['Inference Time'] * (len(cnn)+len(don)+len(em))

df = {'data': all.numpy(), 'group': group_l, 'hue': group_hue}

#plt.clf()
plt.yscale('symlog')
plt.grid()
fig, axes = plt.subplots(2, 2)
sns_plot = sns.violinplot(data=df, x="group", y="data", hue = 'hue', split=True, palette = "Set2", ax=axes[0,0])
#fig = sns_plot.get_figure()
#fig.savefig('/scratch/xx84/girsanov/fbsde/bsb/figure/violin_cnn.png')

#########################################################################################################

cnn = torch.zeros(41, 25)
don = torch.zeros(41, 25)
em = torch.zeros(41, 25)
for i in range(1,25):
    num_time = i * 10
    with open('/scratch/xx84/girsanov/fbsde/hjb/result/'+str(dim)+'_'+str(num_time)+'_hjb_cnn_loss.npy', 'rb') as f:
        cnn_current = torch.Tensor(np.load(f))
    with open('/scratch/xx84/girsanov/fbsde/hjb/result/'+str(dim)+'_'+str(num_time)+'_hjb_don_loss.npy', 'rb') as f:
        don_current = torch.Tensor(np.load(f))
    with open('/scratch/xx84/girsanov/fbsde/hjb/result/'+str(dim)+'_'+str(num_time)+'_hjb_em_loss.npy', 'rb') as f:
        em_current = torch.Tensor(np.load(f))
    cnn[:,i-1] = cnn_current[0,:]
    don[:,i-1] = don_current[0,:]
    em[:,i-1] = em_current[0,:]

cnn = torch.Tensor(cnn.numpy()[np.where(cnn.numpy()!=0)])
don = torch.Tensor(don.numpy()[np.where(don.numpy()!=0)])
em = torch.Tensor(em.numpy()[np.where(em.numpy()!=0)])

cnn_t = torch.zeros(41, 25)
don_t = torch.zeros(41, 25)
em_t = torch.zeros(41, 25)
for i in range(1,25):
    num_time = i * 10
    with open('/scratch/xx84/girsanov/fbsde/hjb/result/'+str(dim)+'_'+str(num_time)+'_hjb_cnn_time.npy', 'rb') as f:
        cnn_current_t = torch.Tensor(np.load(f))
    with open('/scratch/xx84/girsanov/fbsde/hjb/result/'+str(dim)+'_'+str(num_time)+'_hjb_don_time.npy', 'rb') as f:
        don_current_t = torch.Tensor(np.load(f))
    with open('/scratch/xx84/girsanov/fbsde/hjb/result/'+str(dim)+'_'+str(num_time)+'_hjb_em_time.npy', 'rb') as f:
        em_current_t = torch.Tensor(np.load(f))
    cnn_t[:,i-1] = cnn_current_t[0,:]
    don_t[:,i-1] = don_current_t[0,:]
    em_t[:,i-1] = em_current_t[0,:]

cnn = torch.Tensor(cnn.numpy()[np.where(cnn.numpy()!=0)])
don = torch.Tensor(don.numpy()[np.where(don.numpy()!=0)])
em = torch.Tensor(em.numpy()[np.where(em.numpy()!=0)])

cnn_t = torch.Tensor(cnn_t.numpy()[np.where(cnn_t.numpy()!=0)])
don_t = torch.Tensor(don_t.numpy()[np.where(don_t.numpy()!=0)])
em_t = torch.Tensor(em_t.numpy()[np.where(em_t.numpy()!=0)])

loss = torch.concatenate((cnn,don,em),dim=0)
time = torch.concatenate((cnn_t,don_t,em_t),dim=0)
all = torch.concatenate((loss,time),dim=0)

group_l = ['NGO'] * len(cnn) + ['DeepONet'] * len(don) + ['Euler-Maruyama'] * len(em) +['NGO'] * len(cnn) + ['DeepONet'] * len(don) + ['Euler-Maruyama'] * len(em)
group_hue = ['Normalized Error'] * (len(cnn)+len(don)+len(em)) + ['Inference Time'] * (len(cnn)+len(don)+len(em))

df = {'data': all.numpy(), 'group': group_l, 'hue': group_hue}

#plt.clf()
plt.yscale('symlog')
plt.grid()
sns_plot = sns.violinplot(data=df, x="group", y="data", hue = 'hue', split=True, palette = "Set2", ax=axes[0,1])

#########################################################################################################

cnn = torch.zeros(83, 25)
don = torch.zeros(83, 25)
em = torch.zeros(83, 25)
for i in range(1,25):
    num_time = i * 10
    with open('/scratch/xx84/girsanov/fk/bs/result/'+str(dim)+'_'+str(num_time)+'_bs_cnn_loss.npy', 'rb') as f:
        cnn_current = torch.Tensor(np.load(f))
    with open('/scratch/xx84/girsanov/fk/bs/result/'+str(dim)+'_'+str(num_time)+'_bs_don_loss.npy', 'rb') as f:
        don_current = torch.Tensor(np.load(f))
    with open('/scratch/xx84/girsanov/fk/bs/result/'+str(dim)+'_'+str(num_time)+'_bs_em_loss.npy', 'rb') as f:
        em_current = torch.Tensor(np.load(f))
    cnn[:,i-1] = cnn_current[0,:]
    don[:,i-1] = don_current[0,:]
    em[:,i-1] = em_current[0,:]

cnn = torch.Tensor(cnn.numpy()[np.where(cnn.numpy()!=0)])
don = torch.Tensor(don.numpy()[np.where(don.numpy()!=0)])
em = torch.Tensor(em.numpy()[np.where(em.numpy()!=0)])

cnn_t = torch.zeros(83, 25)
don_t = torch.zeros(83, 25)
em_t = torch.zeros(83, 25)
for i in range(1,25):
    num_time = i * 10
    with open('/scratch/xx84/girsanov/fk/bs/result/'+str(dim)+'_'+str(num_time)+'_bs_cnn_time.npy', 'rb') as f:
        cnn_current_t = torch.Tensor(np.load(f))
    with open('/scratch/xx84/girsanov/fk/bs/result/'+str(dim)+'_'+str(num_time)+'_bs_don_time.npy', 'rb') as f:
        don_current_t = torch.Tensor(np.load(f))
    with open('/scratch/xx84/girsanov/fk/bs/result/'+str(dim)+'_'+str(num_time)+'_bs_em_time.npy', 'rb') as f:
        em_current_t = torch.Tensor(np.load(f))
    cnn_t[:,i-1] = cnn_current_t[0,:]
    don_t[:,i-1] = don_current_t[0,:]
    em_t[:,i-1] = em_current_t[0,:]

cnn = torch.Tensor(cnn.numpy()[np.where(cnn.numpy()!=0)])
don = torch.Tensor(don.numpy()[np.where(don.numpy()!=0)])
em = torch.Tensor(em.numpy()[np.where(em.numpy()!=0)])

cnn_t = torch.Tensor(cnn_t.numpy()[np.where(cnn_t.numpy()!=0)])
don_t = torch.Tensor(don_t.numpy()[np.where(don_t.numpy()!=0)])
em_t = torch.Tensor(em_t.numpy()[np.where(em_t.numpy()!=0)])

loss = torch.concatenate((cnn,don,em),dim=0)
time = torch.concatenate((cnn_t,don_t,em_t),dim=0)
all = torch.concatenate((loss,time),dim=0)

group_l = ['NGO'] * len(cnn) + ['DeepONet'] * len(don) + ['Euler-Maruyama'] * len(em) +['NGO'] * len(cnn) + ['DeepONet'] * len(don) + ['Euler-Maruyama'] * len(em)
group_hue = ['Normalized Error'] * (len(cnn)+len(don)+len(em)) + ['Inference Time'] * (len(cnn)+len(don)+len(em))

df = {'data': all.numpy(), 'group': group_l, 'hue': group_hue}

#plt.clf()
plt.yscale('symlog')
plt.grid()
sns_plot = sns.violinplot(data=df, x="group", y="data", hue = 'hue', split=True, palette = "Set2", ax=axes[1,0])

#########################################################################################################

cnn = torch.zeros(83, 25)
don = torch.zeros(83, 25)
em = torch.zeros(83, 25)
for i in range(1,25):
    num_time = i * 10
    with open('/scratch/xx84/girsanov/fk/ou/result/'+str(dim)+'_'+str(num_time)+'_ou_cnn_loss.npy', 'rb') as f:
        cnn_current = torch.Tensor(np.load(f))
    with open('/scratch/xx84/girsanov/fk/ou/result/'+str(dim)+'_'+str(num_time)+'_ou_don_loss.npy', 'rb') as f:
        don_current = torch.Tensor(np.load(f))
    with open('/scratch/xx84/girsanov/fk/ou/result/'+str(dim)+'_'+str(num_time)+'_ou_em_loss.npy', 'rb') as f:
        em_current = torch.Tensor(np.load(f))
    cnn[:,i-1] = cnn_current[0,:]
    don[:,i-1] = don_current[0,:]
    em[:,i-1] = em_current[0,:]

cnn = torch.Tensor(cnn.numpy()[np.where(cnn.numpy()!=0)])
don = torch.Tensor(don.numpy()[np.where(don.numpy()!=0)])
em = torch.Tensor(em.numpy()[np.where(em.numpy()!=0)])

cnn_t = torch.zeros(83, 25)
don_t = torch.zeros(83, 25)
em_t = torch.zeros(83, 25)
for i in range(1,25):
    num_time = i * 10
    with open('/scratch/xx84/girsanov/fk/ou/result/'+str(dim)+'_'+str(num_time)+'_ou_cnn_time.npy', 'rb') as f:
        cnn_current_t = torch.Tensor(np.load(f))
    with open('/scratch/xx84/girsanov/fk/ou/result/'+str(dim)+'_'+str(num_time)+'_ou_don_time.npy', 'rb') as f:
        don_current_t = torch.Tensor(np.load(f))
    with open('/scratch/xx84/girsanov/fk/ou/result/'+str(dim)+'_'+str(num_time)+'_ou_em_time.npy', 'rb') as f:
        em_current_t = torch.Tensor(np.load(f))
    cnn_t[:,i-1] = cnn_current_t[0,:]
    don_t[:,i-1] = don_current_t[0,:]
    em_t[:,i-1] = em_current_t[0,:]

cnn = torch.Tensor(cnn.numpy()[np.where(cnn.numpy()!=0)])
don = torch.Tensor(don.numpy()[np.where(don.numpy()!=0)])
em = torch.Tensor(em.numpy()[np.where(em.numpy()!=0)])

cnn_t = torch.Tensor(cnn_t.numpy()[np.where(cnn_t.numpy()!=0)])
don_t = torch.Tensor(don_t.numpy()[np.where(don_t.numpy()!=0)])
em_t = torch.Tensor(em_t.numpy()[np.where(em_t.numpy()!=0)])

loss = torch.concatenate((cnn,don,em),dim=0)
time = torch.concatenate((cnn_t,don_t,em_t),dim=0)
all = torch.concatenate((loss,time),dim=0)

group_l = ['NGO'] * len(cnn) + ['DeepONet'] * len(don) + ['Euler-Maruyama'] * len(em) +['NGO'] * len(cnn) + ['DeepONet'] * len(don) + ['Euler-Maruyama'] * len(em)
group_hue = ['Normalized Error'] * (len(cnn)+len(don)+len(em)) + ['Inference Time'] * (len(cnn)+len(don)+len(em))

df = {'data': all.numpy(), 'group': group_l, 'hue': group_hue}

#plt.clf()
plt.yscale('symlog')
plt.grid()
sns_plot = sns.violinplot(data=df, x="group", y="data", hue = 'hue', split=True, palette = "Set2", ax=axes[1,1])

plt.tight_layout()
fig.savefig('/scratch/xx84/girsanov/fbsde/bsb/figure/violin_cnn.png')