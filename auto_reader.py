import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import sys

filenames = sys.argv[1:]

ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)


MODELS = ['wvd', 'learnmorlet']#, 'melspec', 'sinc', 'raw']
LRS = [0.0002]#, 0.005, 0.0005]
RUNS = [0]
name = 'save_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.npz'
BS=16
J=5
Q=8
DN='small'
HOP=128
BINS=1024
DATASET='dyni'
#.format(args.BS, args.option, args.J, args.Q, args.L,    
#args.bins, args.model, args.LR, args.dataset, args.run
T = list()
QQ = list()
for RUN in RUNS:
    for lr in LRS:
        for c, model in enumerate(MODELS):
            if 'wvd' in model:
                L=8
            else:
                L=1
            filename = name.format(BS, model, J, Q, L, BINS, DN, lr, DATASET, RUN)
            f = np.load(filename)

#            train = f['train'].squeeze().mean(1)
            test = f['test']
            valid = f['valid']
            print(train[:5])
            T.append(test[valid[:,1].argmax(), 1])
            QQ.append(valid[:, 1].max())
            print(train.shape, test.shape, valid.shape)
            ax1.plot(valid[:, 1], c='C{}'.format(c), label=model)
            ax2.plot(test[:, 1], c='C{}'.format(c))
#            ax3.plot(train, c='C{}'.format(c))

T = np.array(T).reshape((len(RUNS), len(LRS), len(MODELS)))
QQ = np.array(QQ).reshape((len(RUNS), len(LRS), len(MODELS)))

print(MODELS)
Tp = np.take_along_axis(T, QQ.argmax(1)[:, None, :], 1)
print(Tp.mean(0))
print(Tp.std(0))
sdf
handles, labels = ax1.get_legend_handles_labels()

plt.legend(handles[:len(MODELS)], labels[:len(MODELS)])
plt.show(block=True)
asdf

for name in filenames:
    if '_wvd' in name:
        f = np.load(name, allow_pickle=True)
        filters = f['filter']
        plt.figure()
        for i in range(10):
            plt.subplot(2, 10, 1+i)
            plt.imshow(filters[0][2][i], aspect='auto', cmap='Greys')
            plt.subplot(2, 10, 11+i)
            plt.imshow(filters[-1][2][i], aspect='auto', cmap='Greys')
    elif 'npwvd' in name:
        f = np.load(name, allow_pickle=True)
        filters = f['filter']
        plt.figure()
        for i in range(10):
            plt.subplot(2, 10, 1+i)
            plt.imshow(filters[0][0][i], aspect='auto', cmap='Greys')
            plt.subplot(2, 10, 11+i)
            plt.imshow(filters[-1][0][i], aspect='auto', cmap='Greys')



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
