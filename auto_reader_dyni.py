import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib
matplotlib.use('Agg')

filename = sys.argv[-1]

ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)


MODELS = ['wvd', 'sinc', 'learnmorlet']#'wvd', 'learnmorlet', 'melspec', 'sinc', 'morlet']
<<<<<<< HEAD:auto_reader_bird.py
LRS = [0.0002, 0.005, 0.001]
RUNS = range(10)

name = '/mnt/drive2/rbalSpace/WVD/save_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.npz'
=======
LRS = [0.005, 0.001, 0.0002]#, 0.005, 0.0005]
RUNS = range(8)

name = '/mnt/project2/rb42Data/WVD/save_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.npz'
>>>>>>> 9e0b460850ccf32a8c4b79606028f3fbc6bcdec7:auto_reader_dyni.py
BS=16
J=5
Q=16
DNS=['onelayer_linear_scattering', 'onelayer_nonlinear_scattering',
        'joint_linear_scattering']
HOP=64
BINS=1024
<<<<<<< HEAD:auto_reader_bird.py
DATASET='bird'
=======
DATASET='dyni'
>>>>>>> 9e0b460850ccf32a8c4b79606028f3fbc6bcdec7:auto_reader_dyni.py
#.format(args.BS, args.option, args.J, args.Q, args.L,    
#args.bins, args.model, args.LR, args.dataset, args.run
T = list()
QQ = list()

for DN in DNS:
<<<<<<< HEAD:auto_reader_bird.py
    for c, model in enumerate(MODELS):
        if 'wvd' in model:
            L=6
        else:
            L=0
        for RUN in RUNS:
            for lr in LRS:
=======
    for RUN in RUNS:
        for lr in LRS:
            for c, model in enumerate(MODELS):
                if 'wvd' in model:
                    L=6
                else:
                    L=0
>>>>>>> 9e0b460850ccf32a8c4b79606028f3fbc6bcdec7:auto_reader_dyni.py
                filename = name.format(BS, model, J, Q, L, BINS, DN, lr, DATASET, HOP, RUN)
                f = np.load(filename)
    
    #            train = f['train'].squeeze().mean(1)
                test = f['test']
                valid = f['valid']
<<<<<<< HEAD:auto_reader_bird.py
                T.append(test[valid[:,1].argmax(), 1]*100)
                QQ.append(valid[:, 1].max())
                print(test.shape, valid.shape, model)
=======
                T.append(test[valid[:,1].argmax(), 1])
                QQ.append(valid[:, 1].max())
                print(test.shape, valid.shape)
>>>>>>> 9e0b460850ccf32a8c4b79606028f3fbc6bcdec7:auto_reader_dyni.py
                ax1.plot(valid[:, 1], c='C{}'.format(c), label=model)
                ax2.plot(test[:, 1], c='C{}'.format(c))
#            ax3.plot(train, c='C{}'.format(c))

<<<<<<< HEAD:auto_reader_bird.py
T = np.array(T).reshape((len(DNS), len(MODELS), len(RUNS), len(LRS)))
QQ = np.array(QQ).reshape((len(DNS), len(MODELS), len(RUNS), len(LRS)))

#print(MODELS)
#Tp = np.take_along_axis(T, QQ.argmax(2)[:, :, :, None], 2)
print(np.nanmean(T, 2).transpose([1, 0, 2]).reshape((len(MODELS), -1)))
print(np.nanstd(T, 2).transpose([1, 0, 2]).reshape((len(MODELS), -1)))
=======
T = np.array(T).reshape((len(DNS), len(RUNS), len(LRS), len(MODELS)))
QQ = np.array(QQ).reshape((len(DNS), len(RUNS), len(LRS), len(MODELS)))

print(MODELS)
Tp = np.take_along_axis(T, QQ.argmax(2)[:, :, None, :], 2)
print(Tp.mean(1))
print(Tp.std(1))
>>>>>>> 9e0b460850ccf32a8c4b79606028f3fbc6bcdec7:auto_reader_dyni.py
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
