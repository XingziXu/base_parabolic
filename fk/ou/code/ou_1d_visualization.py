import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import seaborn_image as isns

mpl.rcParams.update({'font.size': 15})

with open('/scratch/xx84/girsanov/fk/ou/result/ou_cnn_1d_vis.npy', 'rb') as f:
    v_cnn = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fk/ou/result/ou_cnn_1d_vis.npy', 'rb') as f:
    v_gt = torch.Tensor(np.load(f))
    
fig, axes = plt.subplots(1, 2)
sns_plot = isns.imgplot(v_cnn, ax=axes[1,1], robust=True, cmap="deep")
sns_plot = isns.imgplot(v_cnn, ax=axes[1,2], robust=True, cmap="deep")

plt.savefig('/scratch/xx84/girsanov/fk/ou/figure/1d_vis_ou.png')
plt.clf()