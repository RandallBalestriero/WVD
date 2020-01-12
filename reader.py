import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

J = 40
BS = 5
sinc_bins = 512
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

colors = ['r', 'g', 'b']

for color, option in zip(colors, ['sinc', 'melspec', 'wvd8']):
    file1 = open('saveit_0.pkl','rb')
    f = np.load('save_bird_{}_{}_{}_{}.npz'.format(BS, option, J, sinc_bins),
                allow_pickle=True)

    train = f['train']
    test = f['test']
    valid = f['valid']
    rep = f['rep']
    filter = f['filter']
    y_valid = f['y_valid']
    y_test = f['y_test']

    y_test = y_test[:len(test[-1][0])]
    y_valid = y_valid[:len(valid[-1][0])]

    for i in range(len(test)):
        test[i] = [roc_auc_score(y_test, test[i][0]), test[i][1].mean()]
    test = np.array(test)

    for i in range(len(valid)):
        valid[i] = [roc_auc_score(y_valid, valid[i][0]), valid[i][1].mean()]
    valid = np.array(valid)

    ax1.plot(valid[:, 0], c=color)
    ax2.plot(test[:, 0], c=color)

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

