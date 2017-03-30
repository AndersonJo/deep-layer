import numpy as np


def sigmoid(z: np.array) -> np.array:
    """
    :param z: sum of (w^T x + b)
    """
    return 1 / (1 + np.e ** (-z))


def dsigmoid(phi: np.array):
    return phi * (1 - phi)


activations = dict(
    sigmoid=sigmoid,
    dsigmoid=dsigmoid
)
