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
from sklearn.metrics import roc_auc_score, accuracy_score

import argparse
import utils
import data_loader

v1 = T.Variable(np.random.randn(40))
v2 = T.Variable(np.random.randn(40))
b = T.expand_dims(T.stack([v1, v2], 1), 1)
J=40
modes=2
M=32
N=512

    # crate the average vectors
mut = T.Variable(0.05 * np.random.randn(J * modes).astype('float32'),
                 name='xvar')
muf = T.Variable(0.05 * np.random.randn(J * modes).astype('float32'), name='yvar')

# create the covariance matrix
cor = T.Variable(0.01 * np.random.randn(J * modes).astype('float32'),
                 name='cor')
sigmat = T.Variable(
    1. +
    0.1 *
    np.random.randn(
        J *
        modes).astype('float32'),
    name='yvar')
sigmaf = T.Variable(0.05 * np.random.randn(J * modes).astype('float32'), name='cor')
# create the mixing coefficients
mixing = T.Variable(np.ones((modes, 1, 1)).astype('float32'))

# now apply our parametrization
xvar = 0.1 + T.abs(sigmat)
yvar = 0.1 + T.abs(sigmaf)
coeff = T.stop_gradient(T.sqrt(xvar * yvar)) * 0.95
cov = T.eye(2)#T.linalg.inv(T.stack([xvar, T.tanh(cor) * coeff,
             #               T.tanh(cor) * coeff, yvar], 1).reshape((J * modes, 2, 2)))

# get the gaussian filters
time = T.linspace(-5, 5, M)
freq = T.linspace(0, J * 10, N)
XX = T.meshgrid(time, freq)
XX = T.random.randn((M * N,))
# T.stack([XX.flatten(), XX.flatten()], 1)
grid = T.random.randn((M * N, 2))
mutt = T.expand_dims(T.stack([mut, muf], 1), 1)
centered = grid - mutt
# asdf
gaussian = T.exp(-(T.matmul(centered, cov)**2).sum(-1))
gaussian_2d = (
    T.abs(mixing) *
    T.reshape(
        gaussian /
        T.linalg.norm(
            gaussian,
            2,
            1,
            keepdims=True),
        (J,
         modes,
         N,
         M))).sum(1)
 

g = symjax.function(outputs=gaussian_2d)
print(g())
print(g())

