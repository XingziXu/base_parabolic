import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib
import seaborn as sns

matplotlib.rcParams.update({'font.size': 20})

with open('/scratch/xx84/girsanov/generative_modeling/fk/result/2dgaussian_trainset_0.npy', 'rb') as f:
    std = np.load(f)
with open('/scratch/xx84/girsanov/generative_modeling/fk/result/2dgaussian_trainset_20.npy', 'rb') as f:
    t1 = np.load(f)
with open('/scratch/xx84/girsanov/generative_modeling/fk/result/2dgaussian_trainset_40.npy', 'rb') as f:
    t2 = np.load(f)
with open('/scratch/xx84/girsanov/generative_modeling/fk/result/2dgaussian_trainset_60.npy', 'rb') as f:
    t3 = np.load(f)
with open('/scratch/xx84/girsanov/generative_modeling/fk/result/2dgaussian_trainset_80.npy', 'rb') as f:
    t4 = np.load(f)
with open('/scratch/xx84/girsanov/generative_modeling/fk/result/2dgaussian_trainset_100.npy', 'rb') as f:
    t5 = np.load(f)

with open('/scratch/xx84/girsanov/generative_modeling/fk/result/2dgaussian_metaset.npy', 'rb') as f:
    gmeta = np.load(f)

#plt.set(xscale="linear", yscale="symlog")
#plt.xscale('symlog')
#plt.yscale('symlog')

fig, ax = plt.subplots()

sns.scatterplot([100000,100000], color='dodgerblue', label='$p_{meta}$',alpha=0.8, s=100)
sns.scatterplot([100000,100000], color='orangered', label='Gaussian',alpha=0.8, s=100)
sns.scatterplot([100000,100000], color='gold', label='Task',alpha=0.8, s=100)

sns.kdeplot(x=gmeta[:,0], y=gmeta[:,1], thresh=0.1, color='blue', cmap="Blues", fill=True, alpha = 0.8, ax=ax)
sns.kdeplot(x=std[:,0], y=std[:,1], thresh=0.1, color='red', cmap="Reds", fill=True, alpha = 0.8, ax=ax)
sns.kdeplot(x=t1[:,0], y=t1[:,1], thresh=0.1,color='gold', fill=True, alpha = 0.3, ax=ax)
sns.kdeplot(x=t2[:,0], y=t2[:,1], thresh=0.1,color='gold', fill=True, alpha = 0.3, ax=ax)
sns.kdeplot(x=t3[:,0], y=t3[:,1], thresh=0.1,color='gold', fill=True, alpha = 0.3, ax=ax)
sns.kdeplot(x=t4[:,0], y=t4[:,1], thresh=0.1,color='gold', fill=True, alpha = 0.3, ax=ax)
sns.kdeplot(x=t5[:,0], y=t5[:,1], thresh=0.1,color='gold', fill=True, alpha = 0.3, ax=ax)

plt.xlim(-5.1,25.1)
plt.ylim(-5.1,25.1)
plt.grid()
plt.legend(loc='upper left')
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines['left'].set_position(('outward', 5))
plt.gca().spines['bottom'].set_position(('outward', 5))
plt.xlabel('$X_0$')
plt.ylabel('$X_1$')
plt.tight_layout()
plt.savefig('/scratch/xx84/girsanov/generative_modeling/fk/figure/2d_kde.pdf')
plt.clf()