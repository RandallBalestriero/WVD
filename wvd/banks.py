from scipy.signal import morlet
from scipy.signal import chirp as schirp
import symjax.tensor as T
import symjax
import numpy as np
from scipy.stats import multivariate_normal


def sinc(J, Q, N):

    # get the center frequencies
    freqs = 2 ** (np.arange(J * Q + 1) / Q)
    freqs = freqs[1:] * 0.5 + freqs[:-1] * 0.5
    freqs = T.concatenate([T.ones((1,)), freqs])

    # parametrize the frequencies
    f0 = T.Variable(T.abs(1 / freqs[1:]), name="f0")
    f1 = T.Variable(T.abs(1 / freqs[:-1]), name="f1")

    # sampled the bandpass filters
    time = T.linspace(-N // 2, N // 2 - 1, N)
    time_matrix = time.reshape((-1, 1))
    sincs = T.signal.sinc_bandpass(time_matrix, f0, f1)

    # apodize
    apod_filters = sincs * T.signal.hanning(N).reshape((-1, 1))

    # normalize
    # normed_filters = apod_filters / T.linalg.norm(
    #     apod_filters, 2, 0, keepdims=True
    # )

    filters = T.transpose(apod_filters)[::-1]

    return filters


def morlet(J, Q, trainable=False):
    B = 6 * 2 ** J
    time = T.arange(B) - B // 2

    scales = T.Variable(
        2 ** (np.arange(J * Q) / Q), name="scales", trainable=trainable,
    )

    scales_ = scales[:, None]

    filters = symjax.tensor.signal.complex_morlet(
        scales_ * 2.5, 0.5 / scales_, time
    )

    #    filters /= T.linalg.norm(filters, 2, 1, keepdims=True)

    return filters[::-1]


def chirp(J, Q, trainable=False):
    scales = T.Variable(2 ** (np.arange(J * Q) / Q), name="scales")

    time = T.linspace(-6, 6, 11 * 2 ** J)
    gaussian = T.exp(-((16 * time / scales[:, None]) ** 2))

    time = T.arange(11 * 2 ** J)
    time2 = T.linspace(0, 2, 11 * 2 ** J)
    chirpness = np.ones(J * Q) * 2
    chirpness[:4] = np.linspace(1, 2, 4)
    chirps = T.exp(1j * (time2 ** chirpness[:, None]) * time / scales[:, None])

    filters = gaussian * chirps

    w_centers = 11 * 2 ** J
    w_centers = (
        np.pi
        * (chirps[:, w_centers] - chirps[:, w_centers - 1])
        / chirps[:, w_centers - 1]
    )
    w_centers = np.pi / scales[::-1]
    gaussians = []
    for m, scale in zip(w_centers.get(), scales.get()):
        print(m)
        maxi = np.sqrt(2 / scale ** 2 * (scale / 120) ** 2)
        gaussians.append(
            multivariate_normal(
                [0, m],
                np.array(
                    [
                        [2.8 / scale ** 2, maxi * 0.6],
                        [maxi * 0.6, (scale / 120) ** 2],
                    ]
                ),
            )
        )
    return filters[::-1], gaussians


def stft(width, J, Q, Fs):
    scales = T.Variable(np.linspace(1 / 2 ** J, 1, J * Q), name="scales")
    w_centers = scales[::-1]

    time = T.linspace(-1, 1, Fs * 2)
    gaussian = T.exp(-(time ** 2) / width ** 2)
    sines = T.exp(1j * time * Fs * np.pi * w_centers[:, None])

    filters = gaussian * sines

    return filters[::-1]


def gaussian2d(N, J, Q, init="gabor", window=6):
    # get the gaussian filters
    M = 3 * 2 ** J
    freq = T.linspace(0, np.pi, N)
    time = T.linspace(-3, 3, M)
    x, y = T.meshgrid(time, freq)
    grid = T.stack([x.flatten(), y.flatten()], 1)

    # create the covariance parameters
    rho = T.Variable(
        0.0001 * np.random.randn(J * Q).astype("float32"), name="cor"
    )
    positions = T.zeros(J * Q)
    if init == "gabor":
        scales = T.Variable(2 ** (np.arange(J * Q) / Q), name="scales")
        mus = T.stack([positions, np.pi / scales], 1)

        eleven = 6 / M
        sigma_t = T.Variable(eleven * scales, name="sigma_t")
        sigma_f = T.Variable(1 / scales, name="sigma_t")
    else:
        # create the mean parameters
        scales = T.Variable(np.linspace(1 / 2 ** J, 1, J * Q), name="scales")

        mus = T.stack([positions, np.pi * scales[::-1]], 1)

        eleven = window / M
        sigma_t = T.Variable(0.2 * T.ones_like(rho) * eleven, name="sigma_t")
        sigma_f = T.Variable(0.18 * T.ones_like(rho) / eleven, name="sigma_t")
    # parametrize the rho parameter to prevent determinant of 0
    rho_ = 0.9 * T.tanh(rho) * (sigma_t ** 2 * sigma_f ** 2)
    # produce the 2x2 covariance matrix
    covs = T.stack([sigma_t ** 2, rho_, rho_, sigma_f ** 2], 1).reshape(
        (-1, 2, 2)
    )
    covs_inv = T.linalg.inv(T.eye(2) * 0.00001 + covs)

    centered = grid - T.expand_dims(mus, 1)
    mcov = T.einsum("knd,kdb,knb->kn", centered, covs_inv, centered)
    gaussian = T.exp(-mcov)
    return gaussian.reshape((J * Q, N, M))
