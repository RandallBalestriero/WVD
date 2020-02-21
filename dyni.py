import time
import pickle
import jax
import numpy as np
import sys
sys.path.insert(0, "../SymJAX")
from scipy.io.wavfile import read
import glob
import symjax
import symjax.tensor as T
from symjax import layers
from symjax.utils import train_test_split
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(False)
# https://github.com/google/jax/blob/master/jax/lib/xla_bridge.py
from jax.lib import xla_client

import argparse
import utils
import data_loader


parse = argparse.ArgumentParser()
parse.add_argument('--option', type=str, choices=['melspec', 'morlet', 'sinc',
                                                'learnmorlet', 'raw', 'rawshort',
                                                'wvd', 'mwvd'])
parse.add_argument('-J', type=int, default=5)
parse.add_argument('-Q', type=int, default=8)
parse.add_argument('--bins', type=int, default=512)
parse.add_argument('-BS', type=int, default=8)
parse.add_argument('-L', type=int, default=0)
parse.add_argument('-LR', type=float, default=0.001)
parse.add_argument('--model', type=str, choices=['onelayer_nonlinear_scattering',
    'onelayer_linear_scattering', 'joint_linear_scattering',
    'joint_nonlinear_scattering'])
parse.add_argument('--hop', type=int, default=0)
parse.add_argument('--epochs', type=int, default=100)
parse.add_argument('--dataset', type=str)
parse.add_argument('--modes', type=int, default=1)
args = parse.parse_args()

if args.hop == 0:
    args.hop = args.bins // 2


# DATASET LOADING
if args.dataset == 'dyni':
    wavs_train, labels_train, wavs_valid, labels_valid, wavs_test, labels_test = data_loader.load_dyni()
    Y = labels_train.max()+1
elif args.dataset == 'usc':
    wavs_train, labels_train, wavs_valid, labels_valid, wavs_test, labels_test = data_loader.load_usc()
    Y = labels_train.shape[1]
elif args.dataset == 'esc':
    wavs_train, labels_train, wavs_valid, labels_valid, wavs_test, labels_test = data_loader.load_esc()
    Y = labels_train.max()+1
elif args.dataset == 'mnist':
    wavs_train, labels_train, wavs_valid, labels_valid, wavs_test, labels_test = data_loader.load_mnist()
    Y = labels_train.max()+1
elif args.dataset == 'gtzan':
    wavs_train, labels_train, wavs_valid, labels_valid, wavs_test, labels_test = data_loader.load_gtzan()
    Y = labels_train.max()+1
elif args.dataset == 'irmas':
    wavs_train, labels_train, wavs_valid, labels_valid, wavs_test, labels_test = data_loader.load_irmas()
    Y = labels_train.max()+1
elif args.dataset == 'tut':
    wavs_train, labels_train, wavs_valid, labels_valid, wavs_test, labels_test = data_loader.load_tut()
    Y = labels_train.max()+1





# COMPUTATIONAL MODEL

# create placeholders
if args.dataset == 'usc':
    label = T.Placeholder((args.BS, Y), 'int32')
else:
    label = T.Placeholder((args.BS,), 'int32')
input = T.Placeholder((args.BS, len(wavs_train[0])), 'float32')
deterministic = T.Placeholder((), 'bool')

layer = utils.create_transform(input, args)

utils.__dict__[args.model](layer, deterministic, Y)


for l in layer:
    print(l.shape, l)

# create loss function and loss
if args.dataset == 'usc':
    loss = symjax.losses.sigmoid_crossentropy_logits(label, layer[-1]).mean()
    indices = T.greater_equal(layer[-1], 0).astype('int')
    accuracy = T.equal(label, indices).astype('float32').mean(0)
else:
    loss = symjax.losses.sparse_crossentropy_logits(label, layer[-1]).mean()
    accuracy = symjax.losses.accuracy(label, layer[-1]).mean()


print('loss', loss, 'accu', accuracy)
var = sum([lay.variables() for lay in layer if isinstance(lay, layers.Layer)],
          [])

lr = symjax.schedules.PiecewiseConstant(args.LR, {int(args.epochs/2): args.LR/3,
                                        int(3*args.epochs/4): args.LR/6})
print('lr', lr)
opt = symjax.optimizers.Adam(loss, var, lr)
updates = opt.updates
for lay in layer:
    if isinstance(lay, layers.Layer):
        updates.update(lay.updates)

# create the functions
train = symjax.function(input, label, deterministic, outputs=loss,
                           updates=updates)
test = symjax.function(input, label, deterministic,
                          outputs=[loss, accuracy])
#get_repr = symjax.function(input, outputs=layer[0])

filename = '/mnt/project2/rb42Data/WVD/save_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_'
filename = filename.format(args.BS, args.option, args.J, args.Q, args.L,
                            args.bins, args.model, args.LR, args.dataset,
                            args.hop)
for run in range(10):
    TRAIN, TEST, VALID, FILTER, REP = [], [], [], [], []
    opt.reset()
    lr.reset()
    for v in var:
        v.reset()

    for epoch in range(args.epochs):

        # train part
        l = list()
        for xx, xy in symjax.utils.batchify(wavs_train, labels_train,
                                        batch_size=args.BS,
                                        option='random_see_all'):
            l.append(train(xx, xy, 0))
        print('FINALtrain', np.mean(l))
        TRAIN.append(np.array(l))

        # valid and get repr
        l = list()
        for x, y in symjax.utils.batchify(wavs_valid, labels_valid,
                                            batch_size=args.BS,
                                            option='continuous'):
            l.append(test(x, y, 1))

        VALID.append(np.array(l).mean(0))
        print('FINALvalid', VALID[-1])

        # test
        l = list()
        for x, y in symjax.utils.batchify(wavs_test, labels_test,
                                            batch_size=args.BS,
                                            option='continuous'):
            l.append(test(x, y, 1))

        TEST.append(np.array(l).mean(0))
        print('FINALtest', TEST[-1])


        # save the file
        if epoch == 0 or epoch == args.epochs -1:
            # save filter parameters
            if 'wvd' in args.option:
                FILTER.append([layer[0]._mu.get({}), layer[0]._sigma.get({}), 
                    layer[0]._cor.get({}), layer[0]._filter.get({})])
            elif 'sinc' == args.option:
                FILTER.append([layer[0]._freq.get({}), layer[0]._filter.get({})])
            elif 'learnmorlet' == args.option:
                FILTER.append([layer[0]._filters.get({}), layer[0]._freqs.get({}),
                                layer[0]._scales.get({})])


            np.savez(filename + str(run) + '.npz', train=TRAIN, test=TEST,
                     valid=VALID, filter=FILTER)
        # update the learning rate
        lr.update()
