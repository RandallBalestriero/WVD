from scipy.signal import morlet
from symjax import layers
import symjax.tensor as T
import symjax
import numpy as np
import numpy
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, "../TheanoXLA")


def get_scaled_freqs(f0, f1, J):
    f0 = np.array(T.signal.freq_to_mel(f0).get({}))
    f1 = np.array(T.signal.freq_to_mel(f1).get({}))
    m = np.linspace(f0, f1, J)
    # first we compute the mel scale center frequencies for
    # initialization
    f_sp = 200.0 / 3

    # And now the nonlinear scale
    # beginning of log region (Hz)
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp   # same (Mels)

    # step size for log region
    logstep = numpy.log(6.4) / 27.0

    # If we have vector data, vectorize
    freqs = min_log_hz * np.exp(logstep * (m - min_log_mel))
    freqs /= freqs.max()
    return freqs    
 

def generate_sinc_filterbank(f0, f1, J, N):

    # get the center frequencies
    freqs = get_scaled_freqs(f0, f1, J + 1)

    # make it with difference and make it a variable
    freqs = np.stack([freqs[:-1], freqs[1:]], 1)
    freqs[:, 1] -= freqs[:, 0]
    freqs = T.Variable(freqs, name='c_freq')

    # parametrize the frequencies
    f0 = T.abs(freqs[:, 0])
    f1 = f0 + T.abs(freqs[:, 1])

    # sampled the bandpass filters
    time = T.linspace(- N // 2, N // 2 - 1, N)
    time_matrix = time.reshape((-1, 1))
    sincs = T.signal.sinc_bandpass(time_matrix, f0, f1)

    # apodize
    apod_filters = sincs * T.signal.hanning(N).reshape((-1, 1))

    # normalize
    normed_filters = apod_filters / T.linalg.norm(apod_filters, 2, 0,
                                                  keepdims=True)

    filters = T.transpose(T.expand_dims(normed_filters, 1), [2, 1, 0])
    return filters, freqs


def generate_learnmorlet_filterbank(N, J, Q):
    freqs = T.Variable(np.ones(J * Q) * 5)
    scales = 2**(np.linspace(-0.5, np.log2(2 * np.pi * np.log2(N)), J * Q))
    scales = T.Variable(scales)

    filters = T.signal.morlet(N, s=0.01+T.abs(scales.reshape((-1, 1))),
                              w=freqs.reshape((-1, 1)))
    filters_norm = filters / T.linalg.norm(filters, 2, 1, keepdims=True)

    return T.expand_dims(filters_norm, 1), freqs, scales


def generate_morlet_filterbank(N, J, Q):
    freqs = np.ones(J * Q,dtype='float32') * 5
    scales = 2**(np.linspace(-0.5, np.log2(2 * np.pi * np.log2(N)), J * Q))
    filters = T.signal.morlet(N, s=0.1 + scales.reshape((-1, 1)).astype('float32'),
                              w=freqs.reshape((-1, 1)).astype('float32'))
    filters_norm = filters / T.linalg.norm(filters, 2, 1, keepdims=True)
    return T.expand_dims(filters_norm, 1)


def generate_gaussian_filterbank(N, M, J, f0, f1, modes=1):

    # gaussian parameters
    freqs = get_scaled_freqs(f0, f1, J)
    freqs *= (J - 1) * 10

    if modes > 1:
        other_modes = np.random.randint(0, J, J * (modes - 1))
        freqs = np.concatenate([freqs, freqs[other_modes]])

    # crate the average vectors
    mu_init = np.stack([freqs, 0.1 * np.random.randn(J * modes)], 1)
    mu = T.Variable(mu_init.astype('float32'), name='mu')

    # create the covariance matrix
    cor = T.Variable(0.01 * np.random.randn(J * modes).astype('float32'),
                     name='cor')

    sigma_init = np.stack([freqs/6, 1.+ 0.01 * np.random.randn(J * modes)], 1)
    sigma = T.Variable(sigma_init.astype('float32'), name='sigma')

    # create the mixing coefficients
    mixing = T.Variable(np.ones((modes, 1, 1)).astype('float32'))

    # now apply our parametrization
    coeff = T.stop_gradient(T.sqrt((T.abs(sigma) + 0.1).prod(1))) * 0.95
    Id = T.eye(2)
    cov = Id * T.expand_dims((T.abs(sigma)+0.1),1) +\
            T.flip(Id, 0) * (T.tanh(cor) * coeff).reshape((-1, 1, 1))
    cov_inv = T.linalg.inv(cov)

    # get the gaussian filters
    time = T.linspace(-5, 5, M)
    freq = T.linspace(0, J * 10, N)
    x, y = T.meshgrid(time, freq)
    grid = T.stack([y.flatten(), x.flatten()], 1)
    centered = grid - T.expand_dims(mu, 1)
    # asdf
    gaussian = T.exp(-(T.matmul(centered, cov_inv)**2).sum(-1))
    norm = T.linalg.norm(gaussian, 2, 1, keepdims=True)
    gaussian_2d = T.abs(mixing) * T.reshape(gaussian / norm, (J, modes, N, M))
    return gaussian_2d.sum(1, keepdims=True), mu, cor, sigma, mixing


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
        layer.append(layers.Lambda(T.expand_dims(layer[-1], 1), T.abs))

    elif args.option == 'morlet':
        filters = generate_morlet_filterbank(args.bins, args.J, args.Q)
        layer = [
            layers.Conv1D(input_r, args.J * args.Q, args.bins,
                          W=filters.real(), trainable_W=False,
                          stride=args.hop, trainable_b=False, pad='SAME')]
        layer.append(layers.Conv1D(input_r, args.J * args.Q, args.bins, 
                     W=filters.imag(), trainable_W=False,
                     stride=args.hop, trainable_b=False, pad='SAME'))
        layer.append(T.sqrt(layer[-1]**2 + layer[-2]**2))
        layer.append(T.expand_dims(layer[-1], 1))

    elif args.option == 'learnmorlet':
        filters, freqs, scales = generate_learnmorlet_filterbank(
            args.bins, args.J, args.Q)

        layer = [
            layers.Conv1D(input_r, args.J * args.Q, args.bins,
                          W=T.real(filters), trainable_W=False,
                          stride=args.hop, trainable_b=False, pad='SAME')]
        layer.append(layers.Conv1D(input_r, args.J * args.Q, args.bins,
                                   W=T.imag(filters), trainable_W=False,
                                   stride=args.hop, trainable_b=False,
                                   pad='SAME'))
        layer[0].add_variable(freqs)
        layer[0].add_variable(scales)
        layer[0]._filters = filters
        layer[0]._scales = scales
        layer[0]._freqs = freqs
        layer.append(T.sqrt(layer[-1]**2 + layer[-2]**2+0.001))
        layer.append(T.expand_dims(layer[-1], 1))

    elif 'wvd' in args.option:
        WVD = T.signal.wvd(input_r, window=args.bins * 2, L=args.L * 2,
                           hop=args.hop, mode='same')
        if args.option == 'wvd':
            modes = 1
        else:
            modes = 3
        filters, mu, cor, sigma, mixing = generate_gaussian_filterbank(
            args.bins, 64, args.J * args.Q, 5, 22050, modes)
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
        layer[-1]._filter = filters
        layer.append(layers.Lambda(layer[-1], T.abs))

    elif args.option == 'sinc':
        filters, freq = generate_sinc_filterbank(
            5, 22050, args.J * args.Q, args.bins)
        layer = [layers.Conv1D(input.reshape((args.BS, 1, -1)),
                               args.J * args.Q, args.bins, W=filters,
                               stride=args.hop, trainable_b=False,
                               trainable_W=False, pad='SAME')]
        layer[-1]._freq = freq
        layer[-1]._filter = filters
        layer[-1].add_variable(freq)
        layer.append(T.expand_dims(layer[-1], 1))
        layer.append(layers.Lambda(layer[-1], T.abs))
    return layer


def medium_model(layer, deterministic, c):
    # then standard deep network
    layer.append(layers.Conv2D(layer[-1], W_shape=(16, 1, 3, 3)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Lambda(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (2, 3)))

    layer.append(layers.Conv2D(layer[-1], W_shape=(32, 16, 3, 3)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Lambda(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (1, 2)))

    layer.append(layers.Conv2D(layer[-1], W_shape=(32, 32, 3, 3)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Lambda(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (2, 3)))

    layer.append(layers.Conv2D(layer[-1], W_shape=(64, 32, 3, 3)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Lambda(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (1, 3)))

    layer.append(layers.Conv2D(layer[-1], W_shape=(64, 64, 3, 3)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Lambda(layer[-1], T.leaky_relu))

    layer.append(layers.Dense(layer[-1], c))
    return layer


def small_model(layer, deterministic, c):
    # then standard deep network
    print(layer[-1])
    layer.append(layers.Conv2D(layer[-1], W_shape=(16, 1, 3, 3)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Lambda(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (3, 3)))

    layer.append(layers.Conv2D(layer[-1], W_shape=(16, 16, 3, 3)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Lambda(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (3, 3)))

    layer.append(layers.Conv2D(layer[-1], W_shape=(16, 16, 3, 3)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Lambda(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (1, 3)))

    layer.append(layers.Dense(T.leaky_relu(layer[-1]), c))
    return layer


def onelayer_nonlinear_scattering(layer, deterministic, c):
    # then standard deep network

    N = layer[-1].shape[0]
    features = T.log(layer[-1].mean(3).reshape([N, -1])+0.1)

    layer.append(layers.Dropout(features, 0.3, deterministic))

    layer.append(layers.Dense(features, 256))
    layer.append(layers.BatchNormalization(layer[-1], [0], deterministic))
    layer.append(layers.Lambda(layer[-1], T.leaky_relu))
    layer.append(layers.Dropout(layer[-1], 0.1, deterministic))

    layer.append(layers.Dense(layer[-1], c))

    return layer

def onelayer_linear_scattering(layer, deterministic, c):
    # then standard deep network

    N = layer[-1].shape[0]
    features = T.log(layer[-1].mean(3).reshape([N, -1])+0.1)
    layer.append(layers.Dropout(features, 0.1, deterministic))
    layer.append(layers.Dense(layer[-1], c))

    return layer



def joint_nonlinear_scattering(layer, deterministic, c):
    # then standard deep network

    layer.append(layers.Conv2D(layer[-1], 64, (5, 5)))
#    layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Lambda(layer[-1], T.abs))

    N = layer[-1].shape[0]
    features = T.log(T.concatenate([layer[-1].mean(3).reshape([N, -1]),
                              layer[-3].mean(3).reshape([N, -1])], 1)+0.1)
    layer.append(layers.Dropout(features, 0.3, deterministic))

    layer.append(layers.Dense(layer[-1], 256))
    layer.append(layers.BatchNormalization(layer[-1], [0], deterministic))
    layer.append(layers.Lambda(layer[-1], T.leaky_relu))
    layer.append(layers.Dropout(layer[-1], 0.3, deterministic))

    layer.append(layers.Dense(layer[-1], c))
    return layer

def joint_linear_scattering(layer, deterministic, c):
    # then standard deep network

    layer.append(layers.Conv2D(T.log(layer[-1] + 0.1), 64, (32, 16)))
    layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Lambda(layer[-1], T.abs))

    N = layer[-1].shape[0]
    features = T.concatenate([layer[-1].mean(3).reshape([N, -1]),
                            T.log(layer[-4]+0.1).mean(3).reshape([N, -1])], 1)
    layer.append(layers.Dropout(features, 0.1, deterministic))

    layer.append(layers.Dense(layer[-1], c))
    return layer


def model_bird(layer, deterministic, c):
    # then standard deep network
    layer.append(layers.Conv2D(layer[-1], W_shape=(16, 1, 3, 3)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Lambda(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (3, 3)))

    layer.append(layers.Conv2D(layer[-1], W_shape=(16, 16, 3, 3)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Lambda(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (2, 3)))

    layer.append(layers.Conv2D(layer[-1], W_shape=(32, 16, 3, 3)))
    layer.append(layers.BatchNormalization(
        layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Lambda(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (1, 2)))
    layer.append(layers.Dropout(layer[-1], 0.3, deterministic))

    layer.append(layers.Dense(layer[-1], 256))
    layer.append(layers.BatchNormalization(layer[-1], [0], deterministic))
    layer.append(layers.Lambda(layer[-1], T.leaky_relu))
    layer.append(layers.Dropout(layer[-1], 0.5, deterministic))

    layer.append(layers.Dense(layer[-1], 32))
    layer.append(layers.BatchNormalization(layer[-1], [0], deterministic))
    layer.append(layers.Lambda(layer[-1], T.leaky_relu))
    layer.append(layers.Dropout(layer[-1], 0.5, deterministic))

    layer.append(layers.Dense(T.leaky_relu(layer[-1]), c))
    return layer
