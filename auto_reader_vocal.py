import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib
matplotlib.use('Agg')
import tabulate
filename = sys.argv[-1]

ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)


MODELS = ['morlet', 'wvd', 'sinc', 'learnmorlet']#'wvd', 'learnmorlet', 'melspec', 'sinc', 'morlet']
LRS = [0.0002, 0.001, 0.005]
RUNS = range(10)

name = '/mnt/docker_backup/rbalStuff/WVD/save_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.npz'
DNS=['onelayer_linear_scattering', 'onelayer_nonlinear_scattering',
        'joint_linear_scattering']
HOP=256 # OR 64 (dyni bird commands) 256 (mnist piece fsd) 512 (piece)
BINS=1024
BS=16
J=5
Q=16
DATASET='fsd'
#.format(args.BS, args.option, args.J, args.Q, args.L,    
#args.bins, args.model, args.LR, args.dataset, args.run
T = list()
QQ = list()

for DN in DNS:
    for RUN in RUNS:
        for lr in LRS:
            for c, model in enumerate(MODELS):
                if 'wvd' in model:
                    L=6
                else:
                    L=0
                filename = name.format(BS, model, J, Q, L, BINS, DN, lr, DATASET, HOP, RUN)
                f = np.load(filename)
    
    #            train = f['train'].squeeze().mean(1)
                test = f['test']
                valid = f['valid']
#                if valid[:, 1].max() < 0.1:
#                    T.append(np.nan)

#                else:
                T.append(test[valid[:,1].argmax(), 1]*100)
                QQ.append(valid[:, 1].max())
                print(test.shape, valid.shape, model, lr, DN)
#                ax1.plot(valid[:, 1], c='C{}'.format(c), label=model)
#                ax2.plot(test[:, 1], c='C{}'.format(c))
#            ax3.plot(train, c='C{}'.format(c))

T = np.array(T).reshape((len(DNS), len(RUNS), len(LRS), len(MODELS)))
T = np.nanmean(T, 1).transpose((0, 2, 1)).reshape((-1, len(LRS))).T
#QQ = np.array(QQ).reshape((len(DNS), len(RUNS), len(LRS), len(MODELS)))

#print(MODELS)
#T = np.take_along_axis(T, QQ.argmax(1)[:, None,:, :], 1).squeeze()
print(tabulate.tabulate(T.round(1), tablefmt='latex'))
#print(T.std(1))
sdf
handles, labels = ax1.get_legend_handles_labels()

#plt.legend(handles[:len(MODELS)], labels[:len(MODELS)])
#plt.show(block=True)
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
