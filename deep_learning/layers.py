import numpy as np


class Layer(object):
    def __init__(self, n_out, activation=None, batch_input_shape=None):
        self.n_out = n_out
        self.activation = activation
        self.batch_input_shape = batch_input_shape

    def compile(self):
        pass