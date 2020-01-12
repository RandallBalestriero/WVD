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
from theanoxla.utils import train_test_split
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(False)
#https://github.com/google/jax/blob/master/jax/lib/xla_bridge.py
from jax.lib import xla_client
from sklearn.metrics import roc_auc_score, accuracy_score

import argparse
import utils

parse = argparse.ArgumentParser()
parse.add_argument('-option', type=str, choices=['melspec', 'spec', 'sinc',
                                                 'wvd8', 'wvd16', 'wvd32'])
parse.add_argument('-J', type=int)
parse.add_argument('-BS', type=int, default=16)
parse.add_argument('-sinc_bins', type=int, default=512)
parse.add_argument('-support', type=int, default=32)
parse.add_argument('-LR', type=float, default=0.001)
args = parse.parse_args()


############### DATASET LOADING

wavs, labels = theanoxla.datasets.load_freefield1010(subsample=2)
wavs /= wavs.max(1, keepdims=True)

# get shape
if 'wvd' in args.option:
    wavs = np.sort(glob.glob('SAVE/train_{}*.npz'.format(int(args.option[-1]))))
    input_shape = (args.BS,) + np.load(wavs[0])['arr_0'].shape
    print(input_shape)
else:
    input_shape = (args.BS, len(wavs[0]))

# split into train valid and test
wavs_train, wavs_test, labels_train, labels_test = train_test_split(wavs,
                                                                    labels,
                                                                    0.33)
wavs_train, wavs_valid, labels_train, labels_valid = train_test_split(wavs_train,
                                                                      labels_train,
                                                                      0.33)


################ COMPUTATIONAL MODEL

# create placeholders
label = T.Placeholder((args.BS,), 'int32')
input = T.Placeholder(input_shape, 'float32')
deterministic = T.Placeholder((1,), 'bool')


# create the first part of the network with custom TF

if args.option == 'spec':
    layer = [T.signal.spectrogram(input.reshape((args.BS, 1, -1)),
                                  window=args.J, hop=192)]
elif args.option == 'melspec':
    layer = [T.signal.melspectrogram(input.reshape((args.BS, 1, -1)),
                                     window=1024, hop=192, n_filter=args.J,
                                     low_freq=10, high_freq=22050,
                                     nyquist=22050)]
elif 'wvd' in args.option:
    NN = args.support
    # hop size
    hop = ((input.shape[2] - NN) // args.J, 6)
    # extract patches
    patches = T.extract_image_patches(input, (NN, NN), hop=hop)
    J = patches.shape[2]
    # gaussian parameters
    cov = T.Variable(np.random.randn(J, 2, 2).astype('float32'), name='mean')
    mean = T.Variable(np.zeros((J, 1, 2)).astype('float32'), name='cov')
    # get the gaussian filters
    filter = utils.gauss_2d(NN, mean, cov.transpose([0, 2, 1]).matmul(cov))
    # apply the filters
    wvd = T.einsum('nkjab,kab->nkj', patches.squeeze(), filter)
    # add the variables
    layer = [layers.Activation(T.expand_dims(wvd, 1), T.abs)]
    layer[-1].add_variable(cov)
    layer[-1].add_variable(mean)
elif args.option == 'sinc':
    # create the varianles
    freq = T.Variable(np.random.randn(args.J, 2), name='c_freq')
    # parametrize the frequencies
    f0 = T.abs(freq[:, 0])
    f1 = f0+T.abs(freq[:, 1])
    # sampled the bandpass filters
    time = T.linspace(-5, 5, args.sinc_bins).reshape((-1, 1))
    filters = T.transpose(T.expand_dims(T.signal.sinc_bandpass(time, f0, f1), 1),
                          [2, 1, 0])
    # apodize
    apod_filters = filters * T.signal.hanning(args.sinc_bins)
    # normalize
    normed_filters = apod_filters/(apod_filters**2.).sum(2, keepdims=True)
    layer = [layers.Conv1D(input.reshape((args.BS, 1, -1)), W=normed_filters,
                           strides=256, n_filters=args.J,
                           filter_length=args.sinc_bins, b=None)]
    layer[-1].add_variable(freq)
    layer.append(T.expand_dims(layer[-1], 1))
    layer.append(layers.Activation(layer[-1], T.abs))




# then standard deep network
layer.append(layers.Activation(layer[-1]+0.1, T.log))
utils.model_bird(layer, deterministic)

for l in layer:
    print(l.shape)

# create loss function and loss
loss = theanoxla.losses.sparse_crossentropy_logits(label, layer[-1]).mean()
accuracy = theanoxla.losses.accuracy(label, layer[-1]).mean()
proba = T.softmax(layer[-1])[:, 1]
var = sum([lay.variables() for lay in layer if isinstance(lay, layers.Layer)],
          [])

lr = theanoxla.optimizers.PiecewiseConstantSchedule(args.LR, {33: args.LR/3,
                                                              66: args.LR/6})
updates, _ = theanoxla.optimizers.Adam(loss, var, lr)
for lay in layer:
    if isinstance(lay, layers.Layer):
        updates.update(lay.updates)

# create the functions
f = theanoxla.function(input, label, deterministic, outputs = [loss, accuracy],
                       updates=updates)
g = theanoxla.function(input, label, deterministic, outputs = [proba, accuracy])
h = theanoxla.function(input, outputs = [layer[0]], backend='cpu')

# loader
if 'wvd' in args.option:
    def loader(f):
        return np.load(f)['arr_0']
    load_func = (loader, None)
else:
    load_func = None


TRAIN, TEST, VALID, FILTER, REP = [], [], [], [], []
for epoch in range(100):

    # train part
    l = list()
    for x, y in theanoxla.utils.batchify(wavs_train, labels_train, batch_size=args.BS,
                                         option='random_see_all', load_func=load_func,
                                         extra_process = 4):
        l.append(f(x, y, 0))
    print('FINALtrain', np.mean(l, 0))
    TRAIN.append(np.array(l))

    # valid and get repr
    l = list()
    C = list()
    cpt = 0
    for x, y in theanoxla.utils.batchify(wavs_valid, labels_valid, batch_size=args.BS,
                                         option='continuous', load_func=load_func,
                                         extra_process = 4):
        a, b = g(x, y, 1)
        C.append(np.array(a))
        l.append(b)
        if cpt < 3:
            REP.append(np.array(h(x)[0]))
            cpt += 1

    VALID.append([np.concatenate(C), np.stack(l)])
    print('FINALvalid', np.mean(VALID[-1][1], 0))

    # test
    l = list()
    C = list()
    for x, y in theanoxla.utils.batchify(wavs_test, labels_test, batch_size=args.BS,
                                         option='continuous', load_func=load_func,
                                         extra_process = 4):
        a, b = g(x, y, 1)
        C.append(np.array(a))
        l.append(np.array(b))

    TEST.append([np.concatenate(C), np.stack(l)])
    print('FINALtest', np.mean(TEST[-1][1], 0))


    if 'wvd' in args.option:
        FILTER.append([mean.get({}), cov.get({}), filter.get({})])
    elif 'sinc' == args.option:
        FILTER.append([freq.get({})])
    else:
        FILTER = []

    np.savez('save_bird_{}_{}_{}_{}.npz'.format(args.BS, args.option, args.J,
                                       args.sinc_bins), train=TRAIN,
             test=TEST, valid=VALID, rep=REP, filter=FILTER,
             y_valid = labels_valid, y_test=labels_test)
    lr.update()
    print(lr.value.get({}))
