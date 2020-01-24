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
import data_loader



parse = argparse.ArgumentParser()
parse.add_argument('--option', type=str, choices=['melspec', 'morlet', 'sinc',
                                                 'raw', 'rawshort', 'wvd'])
parse.add_argument('-J', type=int, default=5)
parse.add_argument('-Q', type=int, default = 8)
parse.add_argument('--bins', type=int, default=512)
parse.add_argument('-BS', type=int, default=8)
parse.add_argument('-L', type=int, default=1)
parse.add_argument('-LR', type=float, default=0.001)
parse.add_argument('--dataset', type=str, default='bird',
                   choices=['bird', 'gtzan', 'ecg'])
parse.add_argument('--model', type=str, default='base')
args = parse.parse_args()


############### DATASET LOADING

if args.dataset == 'bird':
    wavs_train, labels_train, wavs_valid, labels_valid, wavs_test, labels_test = data_loader.load_bird()
elif args.dataset == 'ecg':
    wavs_train, labels_train, wavs_valid, labels_valid, wavs_test, labels_test = data_loader.load_ecg()
elif args.dataset == 'gtzan':
    wavs_train, labels_train, wavs_valid, labels_valid, wavs_test, labels_test = data_loader.load_gtzan()

################ COMPUTATIONAL MODEL

# create placeholders
label = T.Placeholder((args.BS,), 'int32')
input = T.Placeholder((args.BS, len(wavs_train[0])), 'float32')
deterministic = T.Placeholder((1,), 'bool')

layer = utils.create_transform(input, args)
layer.append(layers.Activation(layer[-1]+0.1, T.log))
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

lr = theanoxla.optimizers.PiecewiseConstant(args.LR, {33: args.LR/3,
                                                              66: args.LR/6})
opt = theanoxla.optimizers.Adam(loss, var, lr)
updates = opt.updates
for lay in layer:
    if isinstance(lay, layers.Layer):
        updates.update(lay.updates)

# create the functions
train = theanoxla.function(input, label, deterministic, outputs = [loss],
                           updates=updates)
test = theanoxla.function(input, label, deterministic,
                          outputs = [loss, accuracy, proba])
get_repr = theanoxla.function(input, outputs = [layer[0]])

TRAIN, TEST, VALID, FILTER, REP, PROBA = [], [], [], [], [], []


print(wavs_train.shape)

for epoch in range(100):

    #### train part
    l = list()
    for x, y in theanoxla.utils.batchify(wavs_train, labels_train, batch_size=args.BS,
                                         option='random_see_all'):
        l.append(train(x, y, 0))
        print(l[-1])
    print('FINALtrain', np.mean(l, 0))
    TRAIN.append(np.array(l))

    #### valid and get repr
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
    print('FINALvalid', np.mean(VALID[-1][1], 0))

    #### test
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
    print('FINALtest', np.mean(TEST[-1][1], 0))

    #### save filter parameters
    if 'wvd' == args.option:
        FILTER.append([layer[0]._mean.get({}), layer[0]._cov.get({}), layer[0]._filter.get({})])
    elif 'sinc' == args.option:
        FILTER.append([layer[0]._freq.get({}), layer[0]._filter.get({})])
    else:
        FILTER = []

    #### save the file
    np.savez('save_bird_{}_{}_{}_{}_{}_{}_{}.npz'.format(args.BS, args.option, args.J, args.Q, args.L,
                                       args.bins, args.model), train=TRAIN,
             test=TEST, valid=VALID, filter=FILTER, proba=PROBA,
             y_valid = labels_valid, y_test=labels_test)

    #### update the learning rate
    lr.update()
