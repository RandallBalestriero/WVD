import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import sys

filenames = sys.argv[1:]

ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)

for name in filenames:
    f = np.load(name)

    train = f['train']
    test = f['test']
    valid = f['valid']

    print(train.shape, test.shape, valid.shape)

    ax1.plot(valid[:, 1].mean(1))
    ax2.plot(test[:, 1].mean(1))
    ax3.plot(train.mean((1, 2)))


# plt.show(block=True)


for name in filenames:
    if 'wvd' in name:
        f = np.load(name, allow_pickle=True)
        filters = f['filter']
        plt.figure()
        for i in range(10):
            plt.subplot(2, 10, 1+i)
            plt.imshow(filters[0][2][i], aspect='auto', cmap='Greys')
            plt.subplot(2, 10, 11+i)
            plt.imshow(filters[-1][2][i], aspect='auto', cmap='Greys')

plt.show(block=True)


sdf
#    asdfasdf
#
#
#    rep = f['rep']
#    filter = f['filter']
#    y_valid = f['y_valid']
#    y_test = f['y_test']
#
#    y_test = y_test[:len(test[-1][0])]
#    y_valid = y_valid[:len(valid[-1][0])]
#
#    for i in range(len(test)):
#        test[i] = [roc_auc_score(y_test, test[i][0]), test[i][1].mean()]
#    test = np.array(test)
#
#    for i in range(len(valid)):
#        valid[i] = [roc_auc_score(y_valid, valid[i][0]), valid[i][1].mean()]
#    valid = np.array(valid)
#
#    ax1.plot(valid[:, 0], c=color)
#    ax2.plot(test[:, 0], c=color)

plt.figure()
option = 'wvd8'
f = np.load('save_bird_{}_{}_{}_{}.npz'.format(BS, option, J, sinc_bins),
            allow_pickle=True)
filter = f['filter']


plt.figure()
for i in range(24):
    plt.subplot(6, 4, i + 1)
    plt.imshow(filter[-1][2][i])

plt.figure()
for i in range(24):
    plt.subplot(6, 4, i + 1)
    plt.imshow(filter[0][2][i])
plt.suptitle('init')

plt.show()
