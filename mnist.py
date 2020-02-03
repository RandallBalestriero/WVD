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
# https://github.com/google/jax/blob/master/jax/lib/xla_bridge.py
from jax.lib import xla_client
from sklearn.metrics import roc_auc_score, accuracy_score

import argparse
import utils
import data_loader


parse = argparse.ArgumentParser()
parse.add_argument('--option', type=str, choices=['melspec', 'morlet', 'sinc', 'learnmorlet',
                                                  'raw', 'rawshort', 'wvd', 'npwvd'])
parse.add_argument('-J', type=int, default=5)
parse.add_argument('-Q', type=int, default=8)
parse.add_argument('--bins', type=int, default=512)
parse.add_argument('-BS', type=int, default=8)
parse.add_argument('-L', type=int, default=1)
parse.add_argument('-LR', type=float, default=0.001)
parse.add_argument('--model', type=str, default='base')
parse.add_argument('--run', type=int, default=0)
parse.add_argument('--hop', type=int, default=0)
args = parse.parse_args()

if args.hop == 0:
    args.hop = args.bins // 2


# DATASET LOADING
args.dataset = 'dyni'
wavs_train, labels_train, wavs_valid, labels_valid, wavs_test, labels_test = data_loader.load_mnist()




# COMPUTATIONAL MODEL

# create placeholders
label = T.Placeholder((args.BS,), 'int32')
input = T.Placeholder((args.BS, len(wavs_train[0])), 'float32')
print(input.shape)
deterministic = T.Placeholder((1,), 'bool')

layer = utils.create_transform(input, args)
#print(layer[0].shape)
#layer.append(layers.Activation(layer[-1]+0.1, T.log))
if args.model == 'base':
    utils.model_bird(layer, deterministic, labels_train.max()+1)
elif args.model == 'small':
    utils.small_model_bird(layer, deterministic, labels_train.max()+1)
else:
    utils.scattering_model_bird(layer, deterministic, labels_train.max()+1)


for l in layer:
    print(l.shape)

# create loss function and loss
loss = theanoxla.losses.sparse_crossentropy_logits(label, layer[-1]).mean()
accuracy = theanoxla.losses.accuracy(label, layer[-1]).mean()
proba = T.softmax(layer[-1])[:, 1]
var = sum([lay.variables() for lay in layer if isinstance(lay, layers.Layer)],
          [])

lr = theanoxla.schedules.PiecewiseConstant(args.LR, {45: args.LR/3,
                                                      70: args.LR/6})
opt = theanoxla.optimizers.Adam(loss, var, lr)
updates = opt.updates
for lay in layer:
    if isinstance(lay, layers.Layer):
        updates.update(lay.updates)

# create the functions
train = theanoxla.function(input, label, deterministic, outputs=loss,
                           updates=updates)
test = theanoxla.function(input, label, deterministic,
                          outputs=[loss, accuracy, proba])
get_repr = theanoxla.function(input, outputs=layer[0])

TRAIN, TEST, VALID, FILTER, REP, PROBA = [], [], [], [], [], []


print(wavs_train.shape)

for epoch in range(90):

    # train part
    l = list()
    for x, y in theanoxla.utils.batchify(wavs_train, labels_train, batch_size=args.BS,
                                         option='random_see_all'):
        l.append(train(x, y, 0))
    print('FINALtrain', np.mean(l))
    TRAIN.append(np.array(l))

    # valid and get repr
    l = list()
    p = list()
    C = list()
#    r = list()
    cpt = 0
    for x, y in theanoxla.utils.batchify(wavs_valid, labels_valid, batch_size=args.BS,
                                         option='continuous'):
        a, b, c = test(x, y, 1)
        C.append(a)
        l.append(b)
        p.append(c)
#        if cpt < 3:
#            r.append(np.array(get_repr(x)[0]))
#            cpt += 1

    VALID.append([np.array(C), np.array(l)])
#    REP.append(np.concatenate(r))
    PROBA.append(np.concatenate(p))
    print('FINALvalid', np.mean(VALID[-1][1]))

    # test
    l = list()
    C = list()
    p = list()
    for x, y in theanoxla.utils.batchify(wavs_test, labels_test, batch_size=args.BS,
                                         option='continuous'):
        a, b, c = test(x, y, 1)
        C.append(a)
        l.append(b)
        p.append(c)

    TEST.append([np.array(C), np.array(l)])
    PROBA.append(np.concatenate(p))
    print('FINALtest', np.mean(TEST[-1][1]))

    # save filter parameters
    if 'wvd' == args.option:
        FILTER.append([layer[0]._mean.get({}), layer[0]._cov.get(
            {}), layer[0]._filter.get({})])
    elif 'npwvd' == args.option:
        FILTER.append([layer[0]._filter.get({})])
    elif 'sinc' == args.option:
        FILTER.append([layer[0]._freq.get({}), layer[0]._filter.get({})])
    elif 'learnmorlet' == args.option:
        FILTER.append([layer[0]._filters.get({}), layer[0]._w.get({}), layer[0]._scales.get({})])
    else:
        FILTER = []

    # save the file
    np.savez('save_MNIST_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.npz'.format(args.BS, args.option, args.J, args.Q, args.L,
                                                         args.bins, args.model, args.LR, args.dataset, args.run), train=TRAIN,
             test=TEST, valid=VALID, filter=FILTER, proba=PROBA,
             y_valid=labels_valid, y_test=labels_test)

    # update the learning rate
    lr.update()
