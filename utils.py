import numpy as np
import sys
sys.path.insert(0, "../TheanoXLA")
import theanoxla
import theanoxla.tensor as T
from theanoxla import layers
from scipy.signal import morlet

def morlet2(M, w, s):
    x = T.linspace(-s * 2 * np.pi, s * 2 * np.pi, M)
    output = T.complex(T.cos(w * x), T.sin(w * x))

    output2 = output - T.exp(-0.5 * (w**2))

    return output2 * T.exp(-0.5 * (x**2)) * np.pi**(-0.25)

def freq_to_mel(f, option='linear'):
    # convert frequency to mel with
    if option == 'linear':

        # linear part slope
        f_sp = 200.0 / 3

        # Fill in the log-scale part
        min_log_hz = 1000.0    # beginning of log region (Hz)
        min_log_mel = min_log_hz / f_sp   # same (Mels)
        logstep = numpy.log(6.4) / 27.0    # step size for log region
        mel = min_log_mel + np.log(f / min_log_hz) / logstep
        return np.where(f >= min_log_hz, mel, f/f_sp)
    else:
        return 2595 * T.log10(1+f / 700)



def morlet_filter_bank(N, J, Q):
    scales = 2**(np.linspace(-0.5, N**0.25, J*Q))
    filters = [morlet(N, s=s) for s in scales]
    return np.stack(filters)


def gauss_2d(N, m, S):
    time = T.linspace(-5, 5, N)
    x, y = T.meshgrid(time, time)
    grid = T.stack([x.flatten(), y.flatten()], 1) - m
    gaussian = T.exp(-0.5 * (grid.matmul(S)*grid).sum(-1)).reshape((-1, N, N))
#    deter = T.reshape(T.sqrt(S[:, 0, 0]*S[:, 1, 1]-S[:, 0, 1]*S[:, 1, 0]),
#                      (-1, 1, 1))
    return gaussian# * deter / (3.14159 * 2)


def create_transform(input, args):
    if args.option == 'melspec':
        layer = [T.signal.melspectrogram(input.reshape((args.BS, 1, -1)),
                                         window=args.bins, hop=args.hop, n_filter=args.J*args.Q,
                                         low_freq=3, high_freq=22050,
                                         nyquist=22050)]
    elif args.option == 'raw':
        layer = [layers.Conv1D(input.reshape((args.BS, 1, -1)),
                               strides=args.hop,
                               W_shape=(args.J * args.Q, 1, args.bins), trainable_b=False)]
        layer.append(layers.Activation(T.expand_dims(layer[-1], 1), T.abs))
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
    elif args.option == 'learnmorlet':
        w = T.Variable(np.ones(args.J*args.Q)*5)
        scales = T.Variable(2**(np.linspace(-0.5, args.bins**0.25, args.J*args.Q)))
        filters = T.stack([morlet2(args.bins, w[i], s=scales[i]) for i in range(args.J*args.Q)]).squeeze()
        filters_norm = T.expand_dims(filters/T.norm(filters,2, -1, keepdims=True), 1)

        layer = [layers.Conv1D(input.reshape((args.BS, 1, -1)), W=T.real(filters_norm), trainable_W=False,
                               strides=args.hop, W_shape=filters_norm.shape, trainable_b=False)]
        layer.append(layers.Conv1D(input.reshape((args.BS, 1, -1)), W=T.imag(filters_norm), trainable_W=False,
                                   strides=args.hop, W_shape=filters_norm.shape, trainable_b=False))
        layer[0].add_variable(w)
        layer[0].add_variable(scales)
        layer[0]._filters = filters
        layer[0]._scales = scales
        layer[0]._w = w
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
            coeff = T.stop_gradient(T.sqrt(xvar * yvar)) * 0.95
            cov = T.concatenate(
                [xvar, T.tanh(cor)*coeff, T.tanh(cor)*coeff, yvar], 1).reshape((J+1, 2, 2))
    
            mean = T.Variable(0.1*np.random.randn(J+1, 1,
                                                  2).astype('float32'), name='cov')
            # get the gaussian filters
            w = gauss_2d(B*2, mean, cov)
            filter = w / w.sum((1, 2), keepdims=True)
        else:
            w = T.Variable(np.random.randn(J+1, B*2, B*2))
            filter = T.abs(w) / T.abs(w).sum((1, 2), keepdims=True)
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
            layer[-1].add_variable(w)
    elif args.option == 'sinc':
        # create the varianles
        init = np.linspace(4, 256, args.J*args.Q)
        init = np.stack([init - np.random.rand(args.J*args.Q) * 4,
                         np.random.rand(args.J*args.Q) * 8], 1)
        freq = T.Variable(init/256, name='c_freq')
        # parametrize the frequencies
        f0 = T.abs(freq[:, 0])
        f1 = f0+T.abs(freq[:, 1])
        # sampled the bandpass filters
        time = T.linspace(- args.bins // 2, args.bins //2 -1, args.bins).reshape((-1, 1))
        filters = T.transpose(T.expand_dims(T.signal.sinc_bandpass(time, f0, f1), 1),
                              [2, 1, 0])
        # apodize
        apod_filters = filters * T.signal.hanning(args.bins)
        # normalize
        normed_filters = apod_filters/(apod_filters**2.).sum(2, keepdims=True)
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
