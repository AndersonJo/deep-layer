import numpy as np


def linear(z):
    return z


def dlinear(z):
    batch_size = z.shape[0]
    shapes = [1 for _ in range(len(z.shape) - 1)]
    return np.ones((batch_size, *shapes))


def sigmoid(z: np.array) -> np.array:
    """
    :param z: sum of (w^T x + b)
    """
    return 1 / (1 + np.e ** (-z))


def dsigmoid(phi: np.array):
    return phi * (1 - phi)


activations = dict(
    linear=linear,
    dlinear=dlinear,
    sigmoid=sigmoid,
    dsigmoid=dsigmoid
)
