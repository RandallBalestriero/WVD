import pickle
import numpy as np
import sys
sys.path.insert(0, "../TheanoXLA")
import theanoxla
from theanoxla.utils import train_test_split


def extract_patches(signal, y, length, hop):
    B = (signal.shape[-1] - length) // hop + 1
    windows = np.zeros(signal.shape[:-1] + (B, length))
    for b in range(B):
        windows[:,b,:] = signal[:, hop*b:hop*b + length]
    return windows.reshape((-1, length)), np.repeat(y, B, 0)



def load_bird():
    wavs, labels = theanoxla.datasets.load_freefield1010(subsample=2)
    wavs /= wavs.max(1, keepdims=True)
    wavs, labels = extract_patches(wavs, labels, 2**16, 2**15)
    # split into train valid and test
    print(wavs.shape, labels.shape)
    wavs_train, wavs_test, labels_train, labels_test = train_test_split(wavs,
                                                                        labels,
                                                                        train_size=0.75)
    wavs_train, wavs_valid, labels_train, labels_valid = train_test_split(wavs_train,
                                                                          labels_train,
                                                                      train_size=0.8)
    return wavs_train, labels_train, wavs_valid, labels_valid, wavs_test, labels_test



def load_ecg():
   return 'sdf'




def load_gtzan():
    wavs, labels = theanoxla.datasets.gtzan.load(subsample=2)
    wavs /= wavs.max(1, keepdims=True)
    wavs, labels = extract_patches(wavs, labels, 2**16, 2**15)
    # split into train valid and test
    print(wavs.shape, labels.shape)
    wavs_train, wavs_test, labels_train, labels_test = train_test_split(wavs,
                                                                        labels,
                                                                        train_size=0.75)
    wavs_train, wavs_valid, labels_train, labels_valid = train_test_split(wavs_train,
                                                                          labels_train,
                                                                      train_size=0.8)
    return wavs_train, labels_train, wavs_valid, labels_valid, wavs_test, labels_test
 
