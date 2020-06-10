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

def load_vocal():
    _, _, vowels, data = symjax.datasets.vocalset.load()
    N = 2**17
    for k, datum in enumerate(data):
        datum = datum[::2]
        datum -= datum.mean()
        datum /= datum.max()
        if len(datum) >= N:
            data[k] = datum[:N]
        else:
            P = N - len(datum)
            data[k] = np.pad(datum, [P // 2, P - P // 2])
    data = np.array(data).astype('float32')
    unique_vowels = np.unique(vowels)
    y = [np.nonzero(v == unique_vowels)[0] for v in vowels]
    y = np.array(y).astype('int32').squeeze()

    train, test = train_test_split(data, y, train_size=0.75, seed=1)
    train, valid = train_test_split(*train, train_size=0.8, seed=1)

    return train[0], train[1], valid[0], valid[1], test[0], test[1]



def load_tut():
    train_wavs, train_labels, test_wavs, test_labels, folds = symjax.datasets.TUTacousticscences2017.load()

    train_wavs -= train_wavs.mean(1, keepdims=True)
    train_wavs /= train_wavs.max(1, keepdims=True)

    test_wavs -= test_wavs.mean(1, keepdims=True)
    test_wavs /= test_wavs.max(1, keepdims=True)

    valid_wavs = train_wavs[~folds[:, 1]]
    train_wavs = train_wavs[folds[:, 1]]

    valid_labels = train_labels[~folds[:, 1]]
    train_labels = train_labels[folds[:, 1]]

    return train_wavs[:, :, 0], train_labels, valid_wavs[:, :, 0], valid_labels, test_wavs[:, :, 0], test_labels

def load_commands():
    train_wavs, train_labels = symjax.datasets.speech_commands.load()[:2]

    train_wavs -= train_wavs.mean(1, keepdims=True)
    train_wavs /= (train_wavs.max(1, keepdims=True) + 0.01)

    train, test = train_test_split(train_wavs, train_labels, train_size=0.75,
                                   seed=1)
    train, valid = train_test_split(*train, train_size=0.8, seed=1)
 
    return train[0], train[1], valid[0], valid[1], test[0], test[1]



def load_piece():
    train_wavs, train_labels = symjax.datasets.picidae.load()[:2]
    N = 2**17
    wavs = np.zeros((len(train_wavs), 2 ** 17))
    for i in range(len(train_wavs)):
        print(train_wavs[i].shape)
        if len(train_wavs[i].shape) == 2:
            wavs[i, : len(train_wavs[i])] = train_wavs[i][:N, 0]
        else:
            wavs[i, : len(train_wavs[i])] = train_wavs[i][:N]

    wavs -= wavs.mean(1, keepdims=True)
    wavs /= wavs.max(1, keepdims=True)

    train, test = train_test_split(wavs, train_labels, train_size=0.75,
                                   seed=1)
    train, valid = train_test_split(*train, train_size=0.8, seed=1)
 
    return train[0], train[1], valid[0], valid[1], test[0], test[1]




def load_bird():
    wavs, digits, speakers = symjax.datasets.birdvox_70k.load()
    labels = digits
    wavs -= wavs.mean(1, keepdims=True)
    wavs /= wavs.max(1, keepdims=True)
    print('orig', wavs.shape)
    # split
    train, test = train_test_split(wavs, labels, train_size=0.75,
                                   seed=1)
    train, valid = train_test_split(*train, train_size=0.8, seed=1)
    return train[0], train[1], valid[0], valid[1], test[0], test[1]



def load_fsd():
    data = symjax.datasets.FSDKaggle2018.load()
    N = 2**17
    wavs_train = np.zeros((len(data['wavs_train']), 2 ** 17))
    for i in range(len(data['wavs_train'])):
        w = data['wavs_train'][i]
        wavs_train[i, : len(w)] = w[:N]

    wavs_test = np.zeros((len(data['wavs_test']), 2 ** 17))
    for i in range(len(data['wavs_test'])):
        w = data['wavs_test'][i]
        R = 2 ** 17 - len(w)
        if R > 0:
            wavs_test[i, R // 2: R // 2 + len(w)] = w
        else:
            wavs_test[i, : len(w)] = w[:N]

    labels_train = (np.unique(data['labels_train']) == np.array(data['labels_train'])[:, None]).argmax(1).astype('int32')
    labels_test = (np.unique(data['labels_test']) == np.array(data['labels_test'])[:, None]).argmax(1).astype('int32')

    wavs_train -= wavs_train.mean(1, keepdims=True)
    wavs_train /= wavs_train.max(1, keepdims=True)
    wavs_test -= wavs_test.mean(1, keepdims=True)
    wavs_test /= wavs_test.max(1, keepdims=True)

    train, valid = train_test_split(wavs_train, labels_train, train_size=0.7, seed=1)
    return train[0], train[1], valid[0], valid[1], wavs_test, labels_test



def load_mnist():
    wavs, digits, speakers = symjax.datasets.audiomnist.load()
    labels = digits
#    wavs = wavs[:, ::2]
    wavs -= wavs.mean(1, keepdims=True)
    wavs /= wavs.max(1, keepdims=True)
    print('orig', wavs.shape)
    # split
    train, test = train_test_split(wavs, labels, train_size=0.75, seed=1)
    train, valid = train_test_split(*train, train_size=0.8, seed=1)
 
    return train[0], train[1], valid[0], valid[1], test[0], test[1]


def load_irmas():
    wavs, digits, speakers = symjax.datasets.irmas.load()
    labels = digits
    wavs -= wavs.mean(1, keepdims=True)
    wavs /= wavs.max(1, keepdims=True)
    print('orig', wavs.shape)
    # split
    wavs_train, wavs_test, labels_train, labels_test = train_test_split(wavs,
                                                                        labels,
                                                                        train_size=0.75, seed=1)
    wavs_train, wavs_valid, labels_train, labels_valid = train_test_split(wavs_train,
                                                                          labels_train,
                                                                      train_size=0.8, seed=1)
 
    return wavs_train, labels_train, wavs_valid, labels_valid, wavs_test, labels_test



def load_usc():
    wavs, labels = symjax.datasets.urban.load()
    wavs -= wavs.mean(1, keepdims=True)
    wavs /= wavs.max(1, keepdims=True)
    wavs = wavs[:, ::2]
    print('orig', wavs.shape)
    print(labels.shape)
    # split
    wavs_train, wavs_test, labels_train, labels_test = train_test_split(wavs,
                                                                        labels,
                                                                        train_size=0.75, seed=1)
    wavs_train, wavs_valid, labels_train, labels_valid = train_test_split(wavs_train,
                                                                          labels_train,
                                                                      train_size=0.8, seed=1)
 
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
    origin = '../DOCC10_train/DOCC10_train/'
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
    x_train /= (np.abs(x_train).max(1, keepdims=True) + 0.1)
    y_train = np.array(y_train).astype('int32')
    train, test = train_test_split(x_train, y_train, train_size=0.75, stratify=y_train, seed=1)
    print('after', train[0].shape)
    train, valid = train_test_split(*train, train_size=0.8, stratify=train[1], seed=1)
    print('after', train[0].shape)
    return train[0], train[1], valid[0], valid[1], test[0], test[1]



def load_esc():
    wavs, fine, coarse, _, esc10 = symjax.datasets.esc.load()
    wavs = wavs[np.nonzero(esc10)[0]]
    fine = fine[np.nonzero(esc10)[0]]
    fine = (np.unique(fine) == fine[:, None]).argmax(1).astype('int32')
    wavs -= wavs.mean(1, keepdims=True)
    wavs /= np.abs(wavs).max(1, keepdims=True)
    wavs = wavs[:, ::2]
    labels = fine
    print('origin', wavs.shape)
    
    # split into train valid and test
    print(wavs.shape, labels.shape)
    train, test = train_test_split(wavs, labels, train_size=0.75, stratify=labels, seed=1)
    print('after', train[0].shape)
    train, valid = train_test_split(*train, train_size=0.8, stratify=train[1], seed=1)
    return train[0], train[1], valid[0], valid[1], test[0], test[1]
 

def load_gtzan():
    wavs, labels = symjax.datasets.gtzan.load()
    wavs -= wavs.mean(1, keepdims=True)
    wavs /= np.abs(wavs).max(1, keepdims=True)
    wavs = wavs[:, ::2]
    print('origin', wavs.shape)
    
    # split into train valid and test
    train, test = train_test_split(wavs, labels, train_size=0.75, stratify=labels, seed=1)
    train_valid = train_test_split(*train, train_size=0.8, stratify=train[1], seed=1)
    return train[0], train[1], valid[0], valid[1], test[0], test[1]
 
