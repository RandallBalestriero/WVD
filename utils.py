from scipy.signal import morlet
from symjax import layers
import symjax.tensor as T
import symjax
import numpy as np
import numpy
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, "../TheanoXLA")


def get_freqs(f0, f1, J):
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
    return 1 - freqs / freqs.max()
 

def generate_sinc_filterbank(f0, f1, J, N):

    freqs = get_freqs(f0, f1, J)

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


def generate_gaussian_filterbank(N, M, J, f0, f1, modes=1):
    # gaussian parameters

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
    freqs *= ((J - 1) * 10) + 5
    freqs = J * 10 - freqs.astype('float32')
    freqs = np.concatenate(
        [1.8 + (freqs / J)**2, 5 * np.random.rand(J * (modes - 1))])

    # crate the average vectors
    mu_init = np.stack([freqs, 0.05 * np.random.randn(J * modes)], 1)
    mu = T.Variable(mu_init.astype('float32'), name='mu')

    # create the covariance matrix
    cor = T.Variable(0.01 * np.random.randn(J * modes).astype('float32'),
                     name='cor')

    sigma_init = np.stack([1.+0.1 * np.random.randn(J * modes),
                            freqs], 1)
    sigma = T.Variable(sigma_init.astype('float32'), name='sigma')

    # create the mixing coefficients
    mixing = T.Variable(np.ones((modes, 1, 1)).astype('float32'))

    # now apply our parametrization
    coeff = T.stop_gradient(T.sqrt((T.abs(sigma) + 0.1).prod(1))) * 0.95
    Id = T.eye(2)
    cov = Id * T.expand_dims((T.abs(sigma)+0.1),1) + T.flip(Id, 0) * (T.tanh(cor) * coeff).reshape((-1, 1, 1))
    cov_inv = T.linalg.inv(cov)

    # get the gaussian filters
    time = T.linspace(-5, 5, M)
    freq = T.linspace(0, J * 10, N)
    x, y = T.meshgrid(time, freq)
    grid = T.stack([x.flatten(), y.flatten()], 1)
    centered = grid - T.expand_dims(mu, 1)
    # asdf
    gaussian = T.exp(-(T.matmul(centered, cov_inv)**2).sum(-1))
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
    for i in range(10):
        plt.subplot(10, 1, 1+i)
        plt.imshow(gaussian_2d.get({})[i], apsect='auto')
    plt.show()
    return T.expand_dims(gaussian_2d, 1), mu, cor, sigma, mixing


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
        layer = [
            layers.Conv1D(
                input_r,
                W=filters.real,
                trainable_W=False,
                strides=args.hop,
                W_shape=filters.shape,
                trainable_b=False,
                pad='SAME')]
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
        filters, mu, cor, sigma, mixing = generate_gaussian_filterbank(
            args.bins * 2, 32, args.J * args.Q, 5, 22050, args.modes)
        print(WVD)
#        filters=T.random.randn((args.J * args.Q, 1, args.bins*2, 5))
        wvd = T.convNd(WVD, filters)[:, :, 0]
        print('wvd', wvd)
        layer = [layers.Identity(T.expand_dims(wvd, 1))]
        layer[-1].add_variable(mu)
        layer[-1].add_variable(cor)
        layer[-1].add_variable(sigma)
        layer[-1].add_variable(mixing)
        layer[-1]._mu = mu
        layer[-1]._cor = cor
        layer[-1]._sigma = sigma
        layer[-1]._mixing = mixing
        layer.append(layers.Activation(layer[-1], T.abs))

    elif args.option == 'sinc':
        filters, freq = generate_sinc_filterbank(
            5, 22050, args.J * args.Q, args.bins)
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
    layer.append(T.log(layer[-1] + 0.1))
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
    print(layer[-1])
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
