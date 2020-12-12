from symjax.nn import layers
from symjax import nn
import symjax.tensor as T
import symjax
from wvd import banks


def fourier_conv(signal, filters):
    signal_w = T.fft.fft(signal)
    zeros = T.zeros(
        filters.shape[:-1] + (signal.shape[-1] - filters.shape[-1],)
    )
    filters_w = T.fft.fft(T.concatenate([filters, zeros], -1))
    return T.fft.ifft(signal_w * filters_w[None])


def transform(input, args):
    input_r = input.reshape((args.BS, 1, -1))

    if args.option == "melspec":
        output = T.signal.melspectrogram(
            input_r,
            window=args.bins,
            hop=args.hop,
            n_filter=args.J * args.Q,
            low_freq=3,
            high_freq=22050,
            nyquist=22050,
            mode="same",
        )

    elif args.option == "raw":
        output = layers.Conv1D(
            input_r,
            args.J * args.Q,
            5 * 2 ** args.J,
            strides=args.hop,
            trainable_b=False,
            pad="SAME",
        )
        output = T.abs(T.expand_dims(layer[-1], 1))

    elif "morlet" in args.option:
        filters = banks.morlet(
            args.J, args.Q, trainable="learn" in args.option
        )
        output = T.abs(fourier_conv(input_r, filters)[..., :: args.hop])
        output = T.expand_dims(output, 1)

    elif args.option == "sinc":
        N = 15 * 2 ** args.J
        sincs = banks.sinc(args.J, args.Q, N)
        output = T.abs(fourier_conv(input_r, sincs)[..., :: args.hop])
        output = T.expand_dims(output, 1)

    elif "wvd" in args.option:

        WVD = T.signal.wvd(
            input_r,
            window=args.bins * 2,
            L=args.L * 2,
            hop=args.hop,
            mode="same",
        )
        print("WVD", WVD)

        if args.wvdinit == "stftsmall":
            filters = banks.gaussian2d(
                WVD.shape[1], args.J, args.Q, init="stft", window=256
            )
        elif args.wvdinit == "stftlarge":
            filters = banks.gaussian2d(
                WVD.shape[1], args.J, args.Q, init="stft", window=1024
            )
        else:
            filters = banks.gaussian2d(WVD.shape[2], args.J, args.Q)

        print(filters)
        output = fourier_conv(WVD, filters)
        print("output", output)
        output = T.abs(output.sum(2))
        output = T.expand_dims(output, 1)

    return output


def onelayer_nonlinear_scattering(layer, deterministic, c):
    # then standard deep network

    N = layer[-1].shape[0]
    features = T.log(layer[-1].mean(3).reshape([N, -1]) + 0.1)

    layer.append(layers.Dropout(features, 0.3, deterministic))

    layer.append(layers.Dense(features, 256))
    layer.append(layers.BatchNormalization(layer[-1], [0], deterministic))
    layer.append(layers.Lambda(layer[-1], T.leaky_relu))
    layer.append(layers.Dropout(layer[-1], 0.1, deterministic))

    layer.append(layers.Dense(layer[-1], c))

    return layer


def onelayer_linear_scattering(input, deterministic, c):
    # then standard deep network

    N = input.shape[0]
    output = T.log(input.mean(-1).reshape([N, -1]) + 0.1)
    return layers.Dense(output, c)


def joint_linear_scattering(layer, deterministic, c):
    # then standard deep network

    layer.append(layers.Conv2D(T.log(layer[-1] + 0.1), 64, (32, 16)))
    layer.append(
        layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic)
    )
    layer.append(layers.Lambda(layer[-1], T.abs))

    N = layer[-1].shape[0]
    features = T.concatenate(
        [
            layer[-1].mean(3).reshape([N, -1]),
            T.log(layer[-4] + 0.1).mean(3).reshape([N, -1]),
        ],
        1,
    )
    layer.append(layers.Dropout(features, 0.1, deterministic))

    layer.append(layers.Dense(layer[-1], c))
    return layer


def deep_net(input, deterministic, c):

    output = layers.Conv2D(input, 8, (3, 3), b=None, strides=(2, 3))
    output = layers.BatchNormalization(
        output, [1], deterministic=deterministic
    )
    output = nn.relu(output)
    print(output)

    output = layers.Conv2D(output, 8, (3, 3), b=None, strides=(1, 2))
    output = layers.BatchNormalization(
        output, [1], deterministic=deterministic
    )
    output = nn.relu(output)
    print(output)

    output = layers.Conv2D(output, 16, (3, 3), b=None, strides=(1, 2))
    output = layers.BatchNormalization(
        output, [1], deterministic=deterministic
    )
    output = nn.relu(output)
    print(output)

    output = layers.Conv2D(output, 32, (3, 3), b=None, strides=(2, 2))
    output = layers.BatchNormalization(
        output, [1], deterministic=deterministic
    ).mean(-1)
    output = nn.relu(output)
    print(output)

    output = layers.Dropout(output, 0.3, deterministic)
    output = layers.Dense(output, 2 * c)
    output = nn.relu(output)

    return layers.Dense(output, c)
