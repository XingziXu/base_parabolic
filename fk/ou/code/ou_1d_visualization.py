import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib as mpl
import seaborn as sns
#import pandas as pdimport seaborn_image as isns

mpl.rcParams.update({'font.size': 15})

with open('/scratch/xx84/girsanov/fk/ou/result/ou_cnn_1d_vis.npy', 'rb') as f:
    v_cnn = np.load(f)
with open('/scratch/xx84/girsanov/fk/ou/result/ou_gt_1d_vis.npy', 'rb') as f:
    v_gt = np.load(f)
with open('/scratch/xx84/girsanov/fk/ou/result/ou_don_1d_vis.npy', 'rb') as f:
    v_don = np.load(f)
with open('/scratch/xx84/girsanov/fk/ou/result/ou_gir_1d_vis.npy', 'rb') as f:
    v_gir = np.load(f)
with open('/scratch/xx84/girsanov/fk/ou/result/ou_em_1d_vis.npy', 'rb') as f:
    v_em = np.load(f)
with open('/scratch/xx84/girsanov/fk/ou/result/ou_fno_1d_vis.npy', 'rb') as f:
    v_fno = np.load(f)

vmin = min(np.min(v_cnn), np.min(v_gt), np.min(v_don), np.min(v_gir))
vmax = max(np.max(v_cnn), np.max(v_gt), np.max(v_don), np.max(v_gir))

fig, ax = plt.subplots(nrows=3, ncols=2)#,constrained_layout = True)

ax[0,0].imshow(np.rot90(v_gt,1),cmap='YlGnBu',vmin=vmin, vmax=vmax,extent=[0, 1, -1, 1], aspect='auto')
ax[0,0].set_title('Ground Truth')
ax[0,0].spines["top"].set_visible(False)
ax[0,0].spines["right"].set_visible(False)
ax[0,0].spines['left'].set_position(('outward', 5))
ax[0,0].spines['bottom'].set_visible(False)
ax[0,0].set(xticklabels=[])
ax[0,0].set_ylabel('$X$')
ax[0,0].tick_params(bottom=False)

ax[0,1].imshow(np.rot90(v_cnn,1),cmap='YlGnBu',vmin=vmin, vmax=vmax,extent=[0, 1, -1, 1], aspect='auto')
ax[0,1].set_title('NGO')
ax[0,1].spines["top"].set_visible(False)
ax[0,1].spines["right"].set_visible(False)
ax[0,1].spines['left'].set_visible(False)
ax[0,1].spines['bottom'].set_visible(False)
ax[0,1].set(xticklabels=[])
ax[0,1].set(yticklabels=[])
ax[0,1].tick_params(bottom=False, left=False)

ax[1,0].imshow(np.rot90(v_em,1),cmap='YlGnBu',vmin=vmin, vmax=vmax,extent=[0, 1, -1, 1], aspect='auto')
ax[1,0].set_title('Euler-Maruyama')
ax[1,0].spines["top"].set_visible(False)
ax[1,0].spines["right"].set_visible(False)
ax[1,0].spines['left'].set_position(('outward', 5))
ax[1,0].spines['bottom'].set_position(('outward', 5))
ax[1,0].set_xlabel('$t$')
ax[1,0].set_ylabel('$X$')

ax[1,1].imshow(np.rot90(v_gir,1),cmap='YlGnBu',vmin=vmin, vmax=vmax,extent=[0, 1, -1, 1], aspect='auto')
ax[1,1].set_title('Girsanov')
ax[1,1].spines["top"].set_visible(False)
ax[1,1].spines["right"].set_visible(False)
ax[1,1].spines['left'].set_visible(False)
ax[1,1].spines['bottom'].set_position(('outward', 5))
ax[1,1].set_xlabel('$t$')
ax[1,1].set(yticklabels=[])
ax[1,1].tick_params(left=False)

ax[2,0].imshow((v_don),cmap='YlGnBu',vmin=vmin, vmax=vmax,extent=[0, 1, -1, 1], aspect='auto')
ax[2,0].set_title('DeepONet')
ax[2,0].spines["top"].set_visible(False)
ax[2,0].spines["right"].set_visible(False)
ax[2,0].spines['left'].set_visible(False)
ax[2,0].spines['bottom'].set_position(('outward', 5))
ax[2,0].set_xlabel('$t$')
ax[2,0].set(yticklabels=[])
ax[2,0].tick_params(left=False)

ax[2,1].imshow((v_fno),cmap='YlGnBu',vmin=vmin, vmax=vmax,extent=[0, 1, -1, 1], aspect='auto')
ax[2,1].set_title('FNO')
ax[2,1].spines["top"].set_visible(False)
ax[2,1].spines["right"].set_visible(False)
ax[2,1].spines['left'].set_visible(False)
ax[2,1].spines['bottom'].set_position(('outward', 5))
ax[2,1].set_xlabel('$t$')
ax[2,1].set(yticklabels=[])
ax[2,1].tick_params(left=False)

#plt.subplots_adjust(left=0.1,bottom=0.1,right=0.8,top=0.8,wspace=0.4,hspace=0.2)

plt.tight_layout()
plt.savefig('/scratch/xx84/girsanov/fk/ou/figure/1d_vis_ou.png')
plt.clf()
