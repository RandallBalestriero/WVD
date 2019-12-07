import time
import pickle
import jax
import numpy as np
import sys
sys.path.insert(0, "../TheanoXLA")
from scipy.io.wavfile import read
import glob
import theanoxla
import theanoxla.tensor as T
from theanoxla import layers

import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(False)
#https://github.com/google/jax/blob/master/jax/lib/xla_bridge.py
from jax.lib import xla_client
from sklearn.metrics import roc_auc_score, accuracy_score

import argparse

parse = argparse.ArgumentParser()
parse.add_argument('-L', type=int)
args = parse.parse_args()

# variables
L = args.L
BS = 16

# dataset
wavs, labels = theanoxla.datasets.load_freefield1010(subsample=2)
N = len(wavs[0])
##################
# /mnt/storage/rb42Data/
wavs = np.sort(glob.glob('SAVE/train_{}*.npz'.format(L)))

wavs_train, wavs_test, labels_train, labels_test = theanoxla.utils.train_test_split(wavs, labels, 0.33)




# create theano variables
#######################
wav = np.load('SAVE/train_{}_0000.npz'.format(L))['arr_0']
#signal = T.Placeholder((BS,) + wav.shape, 'float32')

# compute the S transform or spectrogram
label = T.Placeholder((BS,), 'int32')
signal = T.Placeholder((BS, N), 'float32')
if L > 0:
    WVD = T.signal.wvd(signal.reshape((BS, 1, -1)), window=1024, L=L, hop=32)
else:
    WVD = T.signal.melspectrogram(signal.reshape((BS, 1, -1)), window=1024, hop=192,
                        n_filter=80, low_freq=10, high_freq=22050,
                        nyquist=22050)
print(wav.shape)
image = T.Placeholder((BS,)+wav.shape, 'float32')
deterministic = T.Placeholder((1,), 'bool')


#print(wav.shape)
#print(signal.shape)
# first layer 2D gaussian or identity
if L > 0:
    NN = 32
    time = T.linspace(-5, 5, NN)
    x, y = T.meshgrid(time, time)
    grid = T.stack([x.flatten(), y.flatten()], 1)
    cov_ = T.Variable(np.eye(2), name='cov')
    cov = cov_.transpose().dot(cov_)
    gaussian = T.exp(-(grid.dot(cov)*grid).sum(1)).reshape((1, 1, NN, NN))
    layer = [layers.Conv2D(image, 1, (NN, NN), strides=(6, 6),
                           W=gaussian, b=None)]
    layer[-1].add_variable(cov_)
    layer.append(layers.Activation(layer[-1], lambda x:T.log(T.abs(x) + 0.01)))
else:
    layer = [layers.Identity(image)]

# then standard deep network
layer.append(layers.Conv2D(layer[-1], 16, (3, 3)))
layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
layer.append(layers.Pool2D(layer[-1], (3, 3)))

layer.append(layers.Conv2D(layer[-1], 16, (3, 3)))
layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
layer.append(layers.Pool2D(layer[-1], (2, 3)))

layer.append(layers.Conv2D(layer[-1], 16, (1, 3)))
layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
layer.append(layers.Pool2D(layer[-1], (1, 2)))

layer.append(layers.Conv2D(layer[-1], 16, (3, 3)))
layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
layer.append(layers.Pool2D(layer[-1], (1, 2)))

layer.append(layers.Dense(layer[-1], 256))
layer.append(layers.BatchNormalization(layer[-1], [0], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
layer.append(layers.Dropout(layer[-1], 0.5, deterministic))

layer.append(layers.Dense(layer[-1], 32))
layer.append(layers.BatchNormalization(layer[-1], [0], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
layer.append(layers.Dropout(layer[-1], 0.5, deterministic))

layer.append(layers.Dense(T.leaky_relu(layer[-1]), 2))

for l in layer:
    print(l.shape)

loss = theanoxla.losses.sparse_crossentropy_logits(label, layer[-1]).mean()
accuracy = theanoxla.losses.accuracy(label, layer[-1]).mean()
proba = T.softmax(layer[-1])[:, 1]
var = sum([lay.variables() for lay in layer], [])

lr = theanoxla.optimizers.PiecewiseConstantSchedule(0.01, {30: 0.005,
                                                            60: 0.002})
updates = theanoxla.optimizers.Adam(loss, var, lr)
for lay in layer:
    updates.update(lay.updates)

f = theanoxla.function(image, label, deterministic, outputs = [loss, accuracy],
                       updates=updates)
g = theanoxla.function(image, label, deterministic, outputs = [proba, accuracy])
h = theanoxla.function(signal, outputs = [WVD], backend='cpu')

# loader
def loader(f):
    return np.load(f)['arr_0']
load_func = (loader, None)

#load_func= None




DATA = []
if L > 0:
    EE = 4
else:
    EE = 0
filename = 'saveit_{}.pkl'.format(L)

TRAIN, TEST, FILTER = [], [], []
for epoch in range(100):

    l = list()
    for x, y in theanoxla.utils.batchify(wavs_train, labels_train, batch_size=BS,
                                         option='random_see_all', load_func=load_func,
                                         extra_process = EE):
        l.append(f(x, y, 0))
    print('FINALtrain', np.mean(l, 0))
    TRAIN.append(np.array(l))

    l = list()
    C = list()
    for x, y in theanoxla.utils.batchify(wavs_test, labels_test, batch_size=BS,
                                         option='continuous', load_func=load_func,
                                         extra_process = EE):
        a, c = g(x, y, 1)
        l.append(a)
        C.append(c)
    l = np.concatenate(l)
    auc = roc_auc_score(labels_test[:len(l)], l)
    aucthresh = roc_auc_score(labels_test[:len(l)], (l > 0.5).astype('int32'))
    TEST.append((np.mean(C), auc, aucthresh))
    print('FINAL', TEST[-1])

    if L > 0:
        FILTER.append([cov.get({}), gaussian.get({})])

    np.savez('saveit_{}.npz'.format(L), train=TRAIN, test=TEST, filter=FILTER)
    lr.update()
    print(lr.value.get({}))
