import pickle
import numpy as np
import sys
sys.path.insert(0, "../TheanoXLA")
import symjax
from symjax.utils import train_test_split


def extract_patches(signal, y, length, hop):
    B = (signal.shape[-1] - length) // hop + 1
    windows = np.zeros(signal.shape[:-1] + (B, length))
    for b in range(B):
        windows[:,b,:] = signal[:, hop*b:hop*b + length]
    return windows.reshape((-1, length)), np.repeat(y, B, 0)


def load_mnist():
    wavs, digits, speakers = symjax.datasets.audiomnist.load()
    labels = digits
    wavs = wavs[:, ::2]
    wavs -= wavs.mean(1, keepdims=True)
    wavs /= wavs.max(1, keepdims=True)
    print('orig', wavs.shape)

    # split
    wavs_train, wavs_test, labels_train, labels_test = train_test_split(wavs,
                                                                        labels,
                                                                        train_size=0.75)
    wavs_train, wavs_valid, labels_train, labels_valid = train_test_split(wavs_train,
                                                                          labels_train,
                                                                      train_size=0.8)
 
    return wavs_train, labels_train, wavs_valid, labels_valid, wavs_test, labels_test



def load_usc():
    wavs, labels = symjax.datasets.urban.load()
    wavs -= wavs.mean(1, keepdims=True)
    wavs /= wavs.max(1, keepdims=True)
    wavs = wavs[:, ::2]
    print('orig', wavs.shape)

    # split
    wavs_train, wavs_test, labels_train, labels_test = train_test_split(wavs,
                                                                        labels,
                                                                        train_size=0.75, stratify=labels)
    wavs_train, wavs_valid, labels_train, labels_valid = train_test_split(wavs_train,
                                                                          labels_train,
                                                                      train_size=0.8, stratify=labels_train)
 
    return wavs_train, labels_train, wavs_valid, labels_valid, wavs_test, labels_test



def load_bird():
    wavs, labels = symjax.datasets.load_freefield1010(subsample=2)
    wavs -= wavs.mean(1, keepdims=True)
    wavs /= wavs.max(1, keepdims=True)
    ind = np.nonzero(labels == 1)[0]
    to_keep = np.nonzero(labels == 0)[0]
    to_keep = np.concatenate([to_keep[:len(ind)], ind])
    wavs = wavs[to_keep]
    labels = labels[to_keep]
    print('orig', wavs.shape)

    # split
    wavs_train, wavs_test, labels_train, labels_test = train_test_split(wavs,
                                                                        labels,
                                                                        train_size=0.75)
    wavs_train, wavs_valid, labels_train, labels_valid = train_test_split(wavs_train,
                                                                          labels_train,
                                                                      train_size=0.8)
 
    return wavs_train, labels_train, wavs_valid, labels_valid, wavs_test, labels_test



def load_ecg():
    train = np.loadtxt('../heartbit/mitbih_train.csv', delimiter=',')
    x_train, y_train = train[:, :-1], train[:, -1]

    test = np.loadtxt('../heartbit/mitbih_test.csv', delimiter=',')
    x_test, y_test = test[:, :-1], test[:, -1]
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.8)
    print(x_train.shape, x_valid.shape, x_test.shape)
    y_train = y_train.astype('int32')
    y_valid = y_valid.astype('int32')
    y_test = y_test.astype('int32')
    print(y_train, y_train.max())
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def load_dyni():
    classes = ['GG', 'GMA', 'LA', 'MB', 'ME', 'PM', 'SSP', 'UDA', 'UDB', 'ZC']
    class2ind = dict(zip(classes, list(range(10))))
    origin = '/home/vrael/DOCC10_train/DOCC10_train/'
    x_train = np.load(origin + 'DOCC10_Xtrain.npy')
    y = np.loadtxt(origin + 'DOCC10_Ytrain.csv',
                   delimiter=',', dtype='str')
    yy = np.loadtxt(origin + 'DOCC10_Xtrain_IDS.csv',
                    delimiter=',', dtype='int32')
    y = y[1:]
    index2id = dict(zip(yy[:, 1], yy[:, 0]))
    id2class = dict(zip(y[:,0], y[:, 1]))
    y_train = list()
    for i in range(len(x_train)):
        idd = index2id[i]
        clas = id2class[str(idd)]
        index = class2ind[clas]
        y_train.append(index)
    x_train /= np.abs(x_train).max(1, keepdims=True)
    y_train = np.array(y_train).astype('int32')
    wavs_train, wavs_test, labels_train, labels_test = train_test_split(x_train,
                                                                        y_train,
                                                                        train_size=0.75, stratify=y_train)
    print('after', wavs_train.shape)
    wavs_train, wavs_valid, labels_train, labels_valid = train_test_split(wavs_train,
                                                                          labels_train,
                                                                      train_size=0.8, stratify=labels_train)
    print('after', wavs_train.shape)
    return wavs_train, labels_train, wavs_valid, labels_valid, wavs_test, labels_test



def load_esc():
    wavs, fine, coarse, _ = symjax.datasets.esc50.load()
    wavs -= wavs.mean(1, keepdims=True)
    wavs /= np.abs(wavs).max(1, keepdims=True)
    wavs = wavs
    labels = fine
    print('origin', wavs.shape)
    
    # split into train valid and test
    print(wavs.shape, labels.shape)
    wavs_train, wavs_test, labels_train, labels_test = train_test_split(wavs,
                                                                        labels,
                                                                        train_size=0.75, stratify=labels)
    print('after', wavs_train.shape)
    wavs_train, wavs_valid, labels_train, labels_valid = train_test_split(wavs_train,
                                                                          labels_train,
                                                                      train_size=0.8, stratify=labels_train)

    return wavs_train, labels_train, wavs_valid, labels_valid, wavs_test, labels_test
 

def load_gtzan():
    wavs, labels = symjax.datasets.gtzan.load()
    wavs -= wavs.mean(1, keepdims=True)
    wavs /= np.abs(wavs).max(1, keepdims=True)
    wavs = wavs[:, ::2]
    print('origin', wavs.shape)
    
    # split into train valid and test
    print(wavs.shape, labels.shape)
    wavs_train, wavs_test, labels_train, labels_test = train_test_split(wavs,
                                                                        labels,
                                                                        train_size=0.75, stratify=labels)
    print('after', wavs_train.shape)
    wavs_train, wavs_valid, labels_train, labels_valid = train_test_split(wavs_train,
                                                                          labels_train,
                                                                      train_size=0.8, stratify=labels_train)

    return wavs_train, labels_train, wavs_valid, labels_valid, wavs_test, labels_test
 
