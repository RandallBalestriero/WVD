import numpy as np
import sys
sys.path.insert(0, "../TheanoXLA")
from scipy.io.wavfile import read

import theanoxla
import theanoxla.tensor as T
from theanoxla import layers

import argparse

parse = argparse.ArgumentParser()
parse.add_argument('-L', type=int)
args = parse.parse_args()

# load dataset
wavs, labels = theanoxla.datasets.load_freefield1010(subsample=2)

# input variables (spectral correlation size
L = args.L

signal = T.Placeholder((wavs.shape[1],), 'float32')

# compute the S transform or spectrogram
if L > 0:
    WVD = T.signal.wvd(signal.reshape((1, 1, -1)), window=1024, L=L, hop=32)
else:
    WVD = T.signal.melspectrogram(signal.reshape((1, 1, -1)), window=1024, hop=192,
                        n_filter=80, low_freq=10, high_freq=22050,
                        nyquist=22050)
print(WVD.shape)
# create the theano function working on cpu
tf_func = theanoxla.function(signal, outputs=[WVD[0, 0]], backend='cpu')

# transform each datum and save it
savename = '/mnt/storage/rb42Data/SAVE/train_{}_{:04d}.npz'
for i, x in enumerate(wavs):
    print(i)
    np.savez_compressed(savename.format(L, i), tf_func(x))


