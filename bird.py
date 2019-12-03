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
wavs, labels, infos = theanoxla.datasets.load_freefield1010(subsample=2)
wavs = np.sort(glob.glob('/mnt/storage/rb42Data/SAVE/train_{}*.npz'.format(L)))
wavs_train, wavs_test, labels_train, labels_test = theanoxla.utils.train_test_split(wavs, labels, 0.33)

# create theano variables
wav = np.load('/mnt/storage/rb42Data/SAVE/train_{}_0.npz'.format(L))['arr_0']
signal = T.Placeholder((BS,) + wav.shape[1:], 'float32')
label = T.Placeholder((BS,), 'int32')
deterministic = T.Placeholder((1,), 'bool')
print('input shape', signal.shape)

# first layer 2D gaussian or identity
if L > 0:
    NN = 32
    time = T.linspace(-5, 5, NN)
    x, y = T.meshgrid(time, time)
    grid = T.stack([x.flatten(), y.flatten()], 1)
    cov_ = T.Variable(np.eye(2), name='cov')
    cov = cov_.transpose().dot(cov_)
    gaussian = T.exp(-(grid.dot(cov)*grid).sum(1)).reshape((1, 1, NN, NN))
    layer = [layers.Conv2D(signal, 1, (NN, NN), strides=(6, 6),
                           W=gaussian, b=None)]
    layer[-1].add_variable(cov_)
    layer.append(layers.Activation(layer[-1], lambda x:T.log(T.abs(x) + 0.01)))
else:
    layer = [layers.Identity(signal)]

# then standard deep network
layer.append(layers.Conv2D(layer[-1], 16, (3, 3)))
layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
layer.append(layers.Pool2D(layer[-1], (3, 3)))

layer.append(layers.Conv2D(layer[-1], 16, (3, 3)))
layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
layer.append(layers.Pool2D(layer[-1], (3, 3)))

layer.append(layers.Conv2D(layer[-1], 32, (3, 3)))
layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
layer.append(layers.Pool2D(layer[-1], (1, 2)))

layer.append(layers.Conv2D(layer[-1], 64, (3, 3)))
layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
#layer.append(layers.Pool2D(layer[-1], (3, 1)))

layer.append(layers.Dense(layer[-1], 256))
layer.append(layers.BatchNormalization(layer[-1], [0], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
layer.append(layers.Dropout(layer[-1], 0.5, deterministic))

layer.append(layers.Dense(layer[-1], 32))
layer.append(layers.BatchNormalization(layer[-1], [0], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
layer.append(layers.Dropout(layer[-1], 0.2, deterministic))

layer.append(layers.Dense(T.relu(layer[-1]), 2))

for l in layer:
    print(l.shape)

loss = theanoxla.losses.sparse_crossentropy_logits(label, layer[-1]).mean()
accuracy = theanoxla.losses.accuracy(label, layer[-1]).mean()
proba = T.softmax(layer[-1])[:, 1]
var = sum([lay.variables() for lay in layer], [])

lr = theanoxla.optimizers.PiecewiseConstantSchedule(0.0001, {30: 0.003,
                                                            60: 0.001})
updates = theanoxla.optimizers.Adam(loss, var, lr)
for lay in layer:
    updates.update(lay.updates)

f = theanoxla.function(signal, label, deterministic, outputs = [loss, accuracy],
                       updates=updates)
g = theanoxla.function(signal, label, deterministic, outputs = [proba, accuracy])


# loader
def loader(f):
    return np.load(f)['arr_0'][0]

load_func = (loader, None)

DATA = []
for epoch in range(100):
    l = list()
    for x, y in theanoxla.utils.batchify(wavs_train, labels_train, batch_size=BS,
                                         option='random_see_all', load_func=load_func,
                                         extra_process = 0):
        l.append(f(x, y, 0))
    DATA.append(l)
    l = list()
    C = list()
    for x, y in theanoxla.utils.batchify(wavs_test, labels_test, batch_size=BS,
                                         option='continuous', load_func=load_func):
        a, c = g(x, y, 1)
        l.append(a)
        C.append(c)
    l = np.concatenate(l)
    DATA.append((np.mean(C), roc_auc_score(labels_test[:len(l)], l)))
    print('FINAL', DATA[-1],
          roc_auc_score(labels_test[:len(l)], (l > 0.5).astype('int32')))
    ff = open('saveit_{}.pkl'.format(L), 'wb')
    pickle.dump(DATA, ff)
    ff.close()
    lr.update()
    print(lr.value.get({}))
