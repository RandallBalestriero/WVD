import numpy as np
import sys
sys.path.insert(0, "../TheanoXLA")
import theanoxla
import theanoxla.tensor as T
from theanoxla import layers
from scipy.signal import morlet


def morlet_filter_bank(N, J, Q):
    scales = 2**(-np.arange(J*Q)/Q)
    filters = [morlet(N, s=s) for s in scales]
    return np.stack(filters)


def gauss_2d(N, m, S):
    time = T.linspace(-5, 5, N)
    x, y = T.meshgrid(time, time)
    grid = T.stack([x.flatten(), y.flatten()], 1) - m
    gaussian = T.exp(-0.5 * (grid.matmul(S)*grid).sum(-1)).reshape((-1, N, N))
    deter = T.reshape(T.sqrt(S[:, 0, 0]*S[:, 1, 1]-S[:, 0, 1]*S[:, 1, 0]),
                      (-1, 1, 1))
    return gaussian * deter / (3.14159 * 2)


def create_transform(input, args):
    if args.option == 'melspec':
        layer = [T.signal.melspectrogram(input.reshape((args.BS, 1, -1)),
                                         window=args.bins, hop=args.hop, n_filter=args.J*args.Q,
                                         low_freq=3, high_freq=22050,
                                         nyquist=22050)]
    elif args.option == 'raw':
        layer = [layers.Conv1D(input.reshape((args.BS, 1, -1)),
                               strides=args.hop,
                               W_shape=(args.J * args.Q, 1, args.bins), b=None)]
        layer.append(layers.Activation(
            layer[-1].reshape(args.BS, q, args.J*args.Q, -1), T.abs))
    elif args.option == 'morlet':
        filters = morlet_filter_bank(args.bins, args.J, args.Q)
        # normalize
        filters = filters/(filters**2).sum(1, keepdims=True)
        filters = np.expand_dims(filters, 1)
        layer = [layers.Conv1D(input.reshape((args.BS, 1, -1)), W=filters.real,trainable_W=False,
                               strides=args.hop, W_shape=filters.shape, trainable_b=False)]
        layer.append(layers.Conv1D(input.reshape((args.BS, 1, -1)), W=filters.imag, trainable_W=False,
                                   strides=args.hop, W_shape=filters.shape, trainable_b=False))
        layer.append(T.sqrt(layer[-1]**2 + layer[-2]**2))
        layer.append(T.expand_dims(layer[-1], 1))
    elif 'wvd' in args.option:
        B = (args.bins//2)//(args.J*args.Q)
        WVD = T.signal.wvd(input.reshape((args.BS, 1, -1)), window=args.bins, L=args.L,
                           hop=(args.hop)//B)

        # extract patches
        patches = T.extract_image_patches(WVD, (B*2, B*2), hop=B)
        J = args.J*args.Q
        # gaussian parameters
        if args.option == 'wvd':
            vart = T.Variable(1+0.01*np.random.randn(J+1, 1).astype('float32'), name='xvar')
            varf = T.Variable(1+0.01*np.random.randn(J+1, 1).astype('float32'), name='yvar')
            cor = T.Variable(
                0.1 * np.random.randn(J+1, 1).astype('float32'), name='cor')
            xvar = 0.1+T.abs(vart)
            yvar = 0.1+T.abs(varf)
            coeff = T.stop_gradient(T.sqrt(xvar * yvar))
            cov = T.concatenate(
                [xvar, T.tanh(cor)*coeff, T.tanh(cor)*coeff, yvar], 1).reshape((J+1, 2, 2))
    
            mean = T.Variable(0.1*np.random.randn(J+1, 1,
                                                  2).astype('float32'), name='cov')
            # get the gaussian filters
            filter = gauss_2d(B*2, mean, cov)
        else:
            filter = T.Variable(np.random.randn(J+1, B*2, B*2))
        # apply the filters
        print(filter.shape, patches.shape)
        wvd = T.einsum('nkjab,kab->nkj', patches.squeeze(), T.abs(filter))
        # add the variables
        layer = [layers.Activation(T.expand_dims(wvd, 1), T.relu)]
        layer[-1]._filter = filter
        if args.option == 'wvd':
            layer[-1]._mean = mean
            layer[-1]._cov = cov
            layer[-1].add_variable(vart)
            layer[-1].add_variable(varf)
            layer[-1].add_variable(cor)
            layer[-1].add_variable(mean)
        else:
            layer[-1].add_variable(filter)
    elif args.option == 'sinc':
        # create the varianles
        freq = T.Variable(np.random.randn(args.J*args.Q, 2), name='c_freq')
        # parametrize the frequencies
        f0 = T.abs(freq[:, 0])
        f1 = f0+T.abs(freq[:, 1])
        # sampled the bandpass filters
        time = T.linspace(-5, 5, args.bins).reshape((-1, 1))
        filters = T.transpose(T.expand_dims(T.signal.sinc_bandpass(time, f0, f1), 1),
                              [2, 1, 0])
        # apodize
        apod_filters = filters * T.signal.hanning(args.bins)
        # normalize
        normed_filters = apod_filters/(apod_filters**2.).sum(2, keepdims=True)
        print(normed_filters.get({}))
        layer = [layers.Conv1D(input.reshape((args.BS, 1, -1)), W=normed_filters,
                               strides=args.hop, trainable_b=False, trainable_W=False,
                               W_shape=(args.Q*args.J, 1, args.bins))]
        layer[-1]._freq = freq
        layer[-1]._filter = normed_filters
        layer[-1].add_variable(freq)
        layer.append(T.expand_dims(layer[-1], 1))
        layer.append(layers.Activation(layer[-1], T.abs))
    return layer


def small_model_bird(layer, deterministic, c):
    # then standard deep network
    layer.append(layers.Conv2D(layer[-1], W_shape=(16, 1, 3, 3)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (3, 3)))

    layer.append(layers.Conv2D(layer[-1], W_shape=(32, 16, 3, 3)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
#    layer.append(layers.Pool2D(layer[-1], (3, 3)))

#    layer.append(layers.Dense(layer[-1], 256))
#    layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
#    layer.append(layers.Activation(layer[-1], T.leaky_relu))
#    layer.append(layers.Dropout(layer[-1], 0.2, deterministic))

    layer.append(layers.Dense(T.leaky_relu(layer[-1]), c))
    return layer


def scattering_model_bird(layer, deterministic, c):
    # then standard deep network
    #layer.append(layers.Pool2D(layer[-1], (1, 5)))
    layer.append(layers.Conv2D(layer[-1], 16, (5, 5)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))

    N = layer[-1].shape[0]
    features = T.concatenate([layer[-1].mean(3).reshape([N, -1]),
                              layer[0].mean(3).reshape([N, -1])], 1)
    print(features.shape)
    layer.append(layers.Dense(features, 256))
    layer.append(layers.BatchNormalization(layer[-1], [0], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dropout(layer[-1], 0.5, deterministic))

    layer.append(layers.Dense(T.leaky_relu(layer[-1]), c))
    return layer


def model_bird(layer, deterministic, c):
    # then standard deep network
    layer.append(layers.Conv2D(layer[-1], W_shape=(16, 1, 3, 3)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (3, 3)))

    layer.append(layers.Conv2D(layer[-1], W_shape=(16, 16, 3, 3)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (2, 3)))

    layer.append(layers.Conv2D(layer[-1], W_shape=(32, 16, 3, 3)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (1, 2)))
    layer.append(layers.Dropout(layer[-1], 0.3, deterministic))

    layer.append(layers.Dense(layer[-1], 256))
    layer.append(layers.BatchNormalization(layer[-1], [0], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dropout(layer[-1], 0.5, deterministic))

    layer.append(layers.Dense(layer[-1], 32))
    layer.append(layers.BatchNormalization(layer[-1], [0], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dropout(layer[-1], 0.5, deterministic))

    layer.append(layers.Dense(T.leaky_relu(layer[-1]), c))
    return layer
