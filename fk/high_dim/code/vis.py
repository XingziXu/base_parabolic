import tensorflow as tf
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

value0 = []
for event in tf.compat.v1.train.summary_iterator('/scratch/xx84/girsanov/fk/high_dim/code/lightning_logs/version_0/events.out.tfevents.1691538222.research-tarokhlab-11.oit.duke.edu.940747.0'):
    for value in event.summary.value:
        if value.tag=='train_loss':
            value0.append(value.simple_value)
        
value1 = []
for event in tf.compat.v1.train.summary_iterator('/scratch/xx84/girsanov/fk/high_dim/code/lightning_logs/version_1/events.out.tfevents.1691539014.research-tarokhlab-11.oit.duke.edu.943801.0'):
    for value in event.summary.value:
        if value.tag=='train_loss':
            value1.append(value.simple_value)
        
value2 = []
for event in tf.compat.v1.train.summary_iterator('/scratch/xx84/girsanov/fk/high_dim/code/lightning_logs/version_2/events.out.tfevents.1691539295.research-tarokhlab-11.oit.duke.edu.944747.0'):
    for value in event.summary.value:
        if value.tag=='train_loss':
            value2.append(value.simple_value)
        
value3 = []
for event in tf.compat.v1.train.summary_iterator('/scratch/xx84/girsanov/fk/high_dim/code/lightning_logs/version_3/events.out.tfevents.1691539581.research-tarokhlab-11.oit.duke.edu.946125.0'):
    for value in event.summary.value:
        if value.tag=='train_loss':
            value3.append(value.simple_value)

value0 = np.expand_dims(np.asarray(value0), axis=1)
value1 = np.expand_dims(np.asarray(value1), axis=1)
value2 = np.expand_dims(np.asarray(value2), axis=1)
value3 = np.expand_dims(np.asarray(value3), axis=1)

values = np.concatenate((value0, value1, value2, value3), axis=1)
value_max = np.max(values, axis=1)
value_min = np.min(values, axis=1)
value_mean = np.mean(values, axis=1)

ax = plt.figure().gca()
ax.plot(value_max, 'k--')
ax.plot(value_min, 'k--')
ax.plot(value_mean, '-', color='royalblue', alpha=0.9)
ax.fill_between(np.linspace(0, len(value_max)-1, len(value_max)), value_min, value_max, color='royalblue', alpha=0.3)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')
plt.savefig('/scratch/xx84/girsanov/fk/high_dim/code/scratch.pdf')