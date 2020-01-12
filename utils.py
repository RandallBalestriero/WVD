import numpy as np
import sys
sys.path.insert(0, "../TheanoXLA")
import theanoxla
import theanoxla.tensor as T
from theanoxla import layers

def gauss_2d(N, m, S):
    time = T.linspace(-5, 5, N)
    x, y = T.meshgrid(time, time)
    grid = T.stack([x.flatten(), y.flatten()], 1) - m
    gaussian = T.exp(-0.5 * (grid.matmul(S)*grid).sum(-1)).reshape((-1, N, N))
    deter = T.reshape(T.sqrt(S[:, 0, 0]*S[:, 1, 1]-S[:, 0, 1]*S[:, 1, 0]),
                      (-1, 1, 1))
    return gaussian * deter / (3.14159 * 2)


def small_model_bird(layer, deterministic):
    # then standard deep network
    layer.append(layers.Conv2D(layer[-1], 16, (3, 3)))
    layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (3, 3)))

    layer.append(layers.Conv2D(layer[-1], 32, (3, 3)))
    layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (3, 3)))

    layer.append(layers.Dense(layer[-1], 256))
    layer.append(layers.BatchNormalization(layer[-1], [0], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dropout(layer[-1], 0.5, deterministic))

    layer.append(layers.Dense(T.leaky_relu(layer[-1]), 2))
    return layer



def scattering_model_bird(layer, deterministic):
    # then standard deep network
    layer.append(layers.Conv2D(layer[-1], 16, (5, 5)))
    layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))

    features = T.concatenate([layer[-1].mean(3).flatten2d(),
                              layer[0].mean(3).flatten2d()], 1)

    layer.append(layers.Dense(features, 256))
    layer.append(layers.BatchNormalization(layer[-1], [0], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dropout(layer[-1], 0.5, deterministic))

    layer.append(layers.Dense(T.leaky_relu(layer[-1]), 2))
    return layer







def model_bird(layer, deterministic):
    # then standard deep network
    layer.append(layers.Conv2D(layer[-1], 16, (3, 3)))
    layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (3, 3)))

    layer.append(layers.Conv2D(layer[-1], 16, (3, 3)))
    layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (2, 3)))

    layer.append(layers.Conv2D(layer[-1], 16, (1, 3)))
    layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (1, 2)))

    layer.append(layers.Conv2D(layer[-1], 16, (3, 3)))
    layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Pool2D(layer[-1], (1, 2)))

    layer.append(layers.Dense(layer[-1], 256))
    layer.append(layers.BatchNormalization(layer[-1], [0], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dropout(layer[-1], 0.5, deterministic))

    layer.append(layers.Dense(layer[-1], 32))
    layer.append(layers.BatchNormalization(layer[-1], [0], deterministic))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dropout(layer[-1], 0.5, deterministic))

    layer.append(layers.Dense(T.leaky_relu(layer[-1]), 2))
    return layer

