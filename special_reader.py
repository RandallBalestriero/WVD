import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib
import matplotlib as mpl


#matplotlib.use('Agg')
max_val = 4
transparency_ticks = 5
colors = [mpl.colors.hsv_to_rgb((0.13, a/transparency_ticks, 1)) 
                  for a in range(transparency_ticks)]
cmap = mpl.colors.ListedColormap(colors)
#norm = mpl.colors.Normalize(vmin=0, vmax=1)


filename = sys.argv[-1]
   
f = np.load(filename, allow_pickle=True)
filters = np.concatenate(f['filter'][-1][3])
filters = np.concatenate([p/p.max() for p in filters[::4]], 1)
print(np.shape(filters))
plt.figure(figsize=(15,10))
plt.imshow(filters, aspect='auto', cmap='jet')
plt.xticks()
plt.yticks()
plt.savefig(filename.split('/')[-1][:-4].replace('.', '-')+'.png')
