from scipy.signal import morlet
from symjax import layers
import symjax.tensor as T
import symjax
import numpy as np
import numpy
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, "../TheanoXLA")


def generate_sinc_filterbank(f0, f1, J, N):

    f = np.linspace(f0, f1, J)
    # first we compute the mel scale center frequencies for
    # initialization
    f_sp = 200.0 / 3
    min_log_hz = 1000.0    # beginning of log region (Hz)
    min_log_mel = min_log_hz / f_sp   # same (Mels)
    logstep = numpy.log(6.4) / 27.0    # step size for log region
    mel = min_log_mel + np.log(f / min_log_hz) / logstep
    freqs = np.where(f >= min_log_hz, mel, f / f_sp)
    # now we initialize the graph
    freqs = 1 - freqs/freqs.max()
    freqs = np.stack([freqs * 0.87, freqs], 1)
    freqs[:, 1] -= freqs[:, 0]
    freq = T.Variable(freqs, name='c_freq')
    # parametrize the frequencies
    f0 = T.abs(freq[:, 0])
    f1 = f0 + T.abs(freq[:, 1])

    # sampled the bandpass filters
    time = T.linspace(- N // 2, N // 2 - 1, N)
    time_matrix = time.reshape((-1, 1))
    sincs = T.signal.sinc_bandpass(time_matrix, f0, f1)
    # apodize
    apod_filters = sincs * T.signal.hanning(N).reshape((-1, 1))
    # normalize
    normed_filters = apod_filters / T.linalg.norm(apod_filters, 2, 0,
                                                  keepdims=True)

    filters = T.transpose(T.expand_dims(sincs, 1), [2, 1, 0])
    return filters, freq


def generate_learnmorlet_filterbank(N, J, Q):
    freqs = T.Variable(np.ones(J * Q) * 5)
    scales = 2**(np.linspace(-0.5, N**0.25, J * Q))
    scales = T.Variable(scales)

    filters = T.signal.morlet(N, s=scales.reshape((-1, 1)),
                              w=freqs.reshape((-1, 1)))
    filters_norm = filters / T.linalg.norm(filters, 2, 1, keepdims=True)
    return T.expand_dims(filters_norm, 1), freqs, scales


def generate_morlet_filterbank(N, J, Q):
    scales = 2**(np.linspace(-0.5, N**0.25, J * Q))
    filters = np.stack([morlet(N, s=s) for s in scales])
    filters = filters / np.linalg.norm(filters, 2, 1, keepdims=True)
    filters = np.expand_dims(filters, 1)
    return filters




def generate_gaussian_filterbank(N, M, J, f0, f1):
# gaussian parameters
    mut = T.Variable(0.01 *  np.random.randn(J).astype('float32'),
                                    name='xvar')

    f = np.linspace(f0, f1, J)
    # first we compute the mel scale center frequencies for
    # initialization
    f_sp = 200.0 / 3
    min_log_hz = 1000.0    # beginning of log region (Hz)
    min_log_mel = min_log_hz / f_sp   # same (Mels)
    logstep = numpy.log(6.4) / 27.0    # step size for log region
    mel = min_log_mel + np.log(f / min_log_hz) / logstep
    freqs = np.where(f >= min_log_hz, mel, f / f_sp)
    freqs /= freqs.max()
    freqs *= ((J-1) * 10) + 5
    freqs = J * 10 - freqs.astype('float32')
    muf = T.Variable(freqs.astype('float32'), name='yvar')
    cor = T.Variable(0. * np.random.randn(J).astype('float32'), name='cor')
    sigmat = T.Variable(1. + 0. * np.random.randn(J).astype('float32'),
                        name='yvar')
    sigmaf = T.Variable(1.8+(freqs/J)**2 + 0.0 * np.random.randn(J).astype('float32'),
                        name='cor')

    # now apply our parametrization
    xvar = 0.1 + T.abs(sigmat)
    yvar = 0.1 + T.abs(sigmaf)
    coeff = T.stop_gradient(T.sqrt(xvar * yvar)) * 0.95
    cov = T.linalg.inv(T.stack([xvar, T.tanh(cor) * coeff,
                   T.tanh(cor) * coeff, yvar], 1).reshape((J, 2, 2)))

    # get the gaussian filters
    time = T.linspace(-5, 5, M)
    freq = T.linspace(0, J * 10, N)
    x, y = T.meshgrid(time, freq)
    grid = T.stack([x.flatten(), y.flatten()], 1)
    centered = grid - T.expand_dims(T.stack([mut, muf], 1), 1)
    gaussian = T.exp(-(T.matmul(centered, cov)**2).sum(-1))
    gaussian_2d = T.reshape(gaussian/T.linalg.norm(gaussian, 2, 1,
                                            keepdims=True), (J, N, M))
#    for i in range(J):
#        plt.subplot(1, J, 1+i)
#        plt.imshow(gaussian_2d[i].get(), aspect='auto')
#    plt.figure()
#    for i in range(J):
#        plt.imshow(gaussian_2d.get().sum(0), aspect='auto')
#    
#    plt.show()
    return T.expand_dims(gaussian_2d,1), mut, muf, cor, sigmat, sigmaf






def create_transform(input, args):
    input_r = input.reshape((args.BS, 1, -1))

    if args.option == 'melspec':
        layer = [
            T.signal.melspectrogram(
                input_r,
                window=args.bins,
                hop=args.hop,
                n_filter=args.J *
                args.Q,
                low_freq=3,
                high_freq=22050,
                nyquist=22050, mode='same')]

    elif args.option == 'raw':
        layer = [layers.Conv1D(input_r, strides=args.hop,
                               W_shape=(args.J * args.Q, 1, args.bins),
                               trainable_b=False, pad='SAME')]
        layer.append(layers.Activation(T.expand_dims(layer[-1], 1), T.abs))

    elif args.option == 'morlet':
        filters = generate_morlet_filterbank(args.bins, args.J, args.Q)
        layer = [layers.Conv1D(input_r, W=filters.real,
                               trainable_W=False, strides=args.hop,
                               W_shape=filters.shape, trainable_b=False, pad='SAME')]
        layer.append(layers.Conv1D(input_r, W=filters.imag, trainable_W=False,
                                   strides=args.hop, W_shape=filters.shape,
                                   trainable_b=False, pad='SAME'))
        layer.append(T.sqrt(layer[-1]**2 + layer[-2]**2))
        layer.append(T.expand_dims(layer[-1], 1))

    elif args.option == 'learnmorlet':
        filters, freqs, scales = generate_learnmorlet_filterbank(
            args.bins, args.J, args.Q)

        layer = [
            layers.Conv1D(
                input_r,
                W=T.real(filters),
                trainable_W=False,
                strides=args.hop,
                W_shape=filters.shape,
                trainable_b=False, pad='SAME')]
        layer.append(layers.Conv1D(input_r, W=T.imag(filters),
                                   trainable_W=False, strides=args.hop,
                                   W_shape=filters.shape,
                                   trainable_b=False, pad='SAME'))
        layer[0].add_variable(freqs)
        layer[0].add_variable(scales)
        layer[0]._filters = filters
        layer[0]._scales = scales
        layer[0]._freqs = freqs
        layer.append(T.sqrt(layer[-1]**2 + layer[-2]**2))
        layer.append(T.expand_dims(layer[-1], 1))

    elif 'wvd' in args.option:
        WVD = T.signal.wvd(input_r, window=args.bins * 4, L=args.L * 2,
                            hop=args.hop, mode='same')
        filters, mut, muf, cor, sigmat, sigmaf = generate_gaussian_filterbank(args.bins*2, 32, args.J*args.Q, 5, 22050)
        wvd = T.convNd(WVD, filters).squeeze()
        layer = [layers.Identity(T.expand_dims(wvd, 1))]
        layer[-1]._filter = filters
        layer[-1]._mut = mut
        layer[-1]._muf = muf
        layer[-1]._cor = cor
        layer[-1]._sigmat = sigmat
        layer[-1]._sigmaf = sigmaf
        layer.append(layers.Activation(layer[-1], T.abs))

    elif args.option == 'sinc':
        filters, freq = generate_sinc_filterbank(5, 22050, args.J*args.Q, args.bins)
        layer = [layers.Conv1D(input.reshape((args.BS, 1, -1)), W=filters,
                               strides=args.hop, trainable_b=False,
                                trainable_W=False,
                                W_shape=(args.Q * args.J, 1, args.bins),
                                pad='SAME')]
        layer[-1]._freq = freq
        layer[-1]._filter = filters
        layer[-1].add_variable(freq)
        layer.append(T.expand_dims(layer[-1], 1))
        layer.append(layers.Activation(layer[-1], T.abs))
    layer.append(T.log(layer[-1]+0.1))
    return layer


def medium_model(layer, deterministic, c):
    # then standard deep network
    layer.append(layers.Conv2D(layer[-1], W_shape=(16, 1, 3, 3)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (2, 3)))

    layer.append(layers.Conv2D(layer[-1], W_shape=(32, 16, 3, 3)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (1, 2)))

    layer.append(layers.Conv2D(layer[-1], W_shape=(32, 32, 3, 3)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (2, 3)))

    layer.append(layers.Conv2D(layer[-1], W_shape=(64, 32, 3, 3)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (1, 3)))


    layer.append(layers.Conv2D(layer[-1], W_shape=(64, 64, 3, 3)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))

    layer.append(layers.Dense(layer[-1], c))
    return layer


def small_model(layer, deterministic, c):
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
    layer.append(layers.Pool2D(layer[-1], (3, 3)))

    layer.append(layers.Conv2D(layer[-1], W_shape=(16, 16, 3, 3)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (1, 3)))

    layer.append(layers.Dense(T.leaky_relu(layer[-1]), c))
    return layer


def scattering_model(layer, deterministic, c):
    # then standard deep network

    layer.append(layers.Conv2D(layer[-1], W_shape=(48, 1, 3, 3)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Activation(layer[-1], T.abs))

    N = layer[-1].shape[0]
    features = T.concatenate([layer[-1].mean(3).reshape([N, -1]),
                              layer[-4].mean(3).reshape([N, -1])], 1)

    layer.append(layers.Dense(features, 256))
    layer.append(layers.BatchNormalization(layer[-1], [0], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))

    layer.append(layers.Dropout(layer[-1], 0.3, deterministic))

    layer.append(layers.Dense(features, 128))
    layer.append(layers.BatchNormalization(layer[-1], [0], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))

    layer.append(layers.Dense(layer[-1], c))
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
