import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib as mpl
import seaborn as sns
#import pandas as pdimport seaborn_image as isns

mpl.rcParams.update({'font.size': 15})

with open('/scratch/xx84/girsanov/fk/ou/result/ou_cnn_1d_vis.npy', 'rb') as f:
    v_cnn = np.load(f)
with open('/scratch/xx84/girsanov/fk/ou/result/ou_cnn_1d_vis.npy', 'rb') as f:
    v_gt = np.load(f)
with open('/scratch/xx84/girsanov/fk/ou/result/ou_don_1d_vis.npy', 'rb') as f:
    v_don = np.load(f)
with open('/scratch/xx84/girsanov/fk/ou/result/ou_gir_1d_vis.npy', 'rb') as f:
    v_gir = np.load(f)
with open('/scratch/xx84/girsanov/fk/ou/result/ou_em_1d_vis.npy', 'rb') as f:
    v_em = np.load(f)


vmin = min(np.min(v_cnn), np.min(v_gt), np.min(v_don), np.min(v_gir))
vmax = max(np.max(v_cnn), np.max(v_gt), np.max(v_don), np.max(v_gir))

fig, ax = plt.subplots(2,2)#,constrained_layout = True)

ax[0,0].imshow(np.rot90(v_gt,2),cmap='YlGnBu',vmin=vmin, vmax=vmax,extent=[-1, 1, 0, 1])
ax[0,0].set_title('Ground Truth')
ax[0,0].spines["top"].set_visible(False)
ax[0,0].spines["right"].set_visible(False)
ax[0,0].spines['left'].set_position(('outward', 5))
ax[0,0].spines['bottom'].set_position(('outward', 5))
ax[0,0].set_xlabel('$X$')
ax[0,0].set_ylabel('$T$')

ax[0,1].imshow(np.rot90(v_cnn,2),cmap='YlGnBu',vmin=vmin, vmax=vmax,extent=[-1, 1, 0, 1])
ax[0,1].set_title('NGO')
ax[0,1].spines["top"].set_visible(False)
ax[0,1].spines["right"].set_visible(False)
ax[0,1].spines['left'].set_position(('outward', 5))
ax[0,1].spines['bottom'].set_position(('outward', 5))
ax[0,1].set_xlabel('$T$')
ax[0,1].set_ylabel('$X$')

ax[1,0].imshow(np.rot90(v_em,2),cmap='YlGnBu',vmin=vmin, vmax=vmax,extent=[-1, 1, 0, 1])
ax[1,0].set_title('Euler-Maruyama')
ax[1,0].spines["top"].set_visible(False)
ax[1,0].spines["right"].set_visible(False)
ax[1,0].spines['left'].set_position(('outward', 5))
ax[1,0].spines['bottom'].set_position(('outward', 5))
ax[1,0].set_xlabel('$X$')
ax[1,0].set_ylabel('$T$')

ax[1,1].imshow(np.rot90(v_gir,2),cmap='YlGnBu',vmin=vmin, vmax=vmax,extent=[-1, 1, 0, 1])
ax[1,1].set_title('Girsanov')
ax[1,1].spines["top"].set_visible(False)
ax[1,1].spines["right"].set_visible(False)
ax[1,1].spines['left'].set_position(('outward', 5))
ax[1,1].spines['bottom'].set_position(('outward', 5))
ax[1,1].set_xlabel('$X$')
ax[1,1].set_ylabel('$T$')

plt.tight_layout()
plt.savefig('/scratch/xx84/girsanov/fk/ou/figure/1d_vis_ou.pdf')
plt.clf()
