import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib as mpl
import seaborn as sns
import pandas as pd

mpl.rcParams.update({'font.size': 15})

with open('/scratch/xx84/girsanov/fbsde/hjb/result/hjb_cnn_1d_vis.npy', 'rb') as f:
    v_cnn = torch.Tensor(np.load(f))
with open('/scratch/xx84/girsanov/fbsde/hjb/result/hjb_cnn_1d_vis.npy', 'rb') as f:
    v_gt = torch.Tensor(np.load(f))
    
plt.subplot(1, 2, 1)
plt.imshow(v_cnn, cmap='jet', aspect='auto')
plt.subplot(1, 2, 2)
plt.imshow(v_gt, cmap='jet', aspect='auto')
plt.tight_layout()
plt.savefig('/scratch/xx84/girsanov/fbsde/hjb/figure/1d_vis_hjb.png')
plt.clf()