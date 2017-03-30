import numpy as np

from deep_learning.activations import activations
from deep_learning.utils import _get_function


class BaseLayer(object):
    def __init__(self,
                 n_in: int,
                 n_out: int,
                 name: str = None,
                 batch_input_shape: tuple = None):
        self.n_in: int = n_in
        self.n_out: int = n_out
        self.name = name
        self.batch_input_shape: tuple = batch_input_shape

        self._is_output_layer = False

    def compile(self):
        raise NotImplementedError('compile method should be implemented')

    def is_input_layer(self):
        raise NotImplementedError('is_input_layer method should be implemented')

    def is_output_layer(self):
        return self._is_output_layer

    def feedforward(self, tensor: np.array):
        raise NotImplementedError('feedforward method should be implemented')

    def __str__(self):
        return f'<Layer {self.name}>'


class Layer(BaseLayer):
    def __init__(self,
                 n_in: int,
                 n_out: int,
                 activation: str = None, dactivation=None,
                 batch_input_shape: tuple = None,
                 name: str = None):
        super(Layer, self).__init__(n_in, n_out,
                                    name=name,
                                    batch_input_shape=batch_input_shape)

        self.activation = _get_function(activations, activation, activation)
        self.dactivation = _get_function(activations, f'd{activation}', lambda x: np.array(1))

        self.w: np.array = None
        self.b: np.array = None

        self.update_w = 0
        self.update_b = 0

        # Shape
        self._shape = None

    def compile(self):
        self.w = np.random.randn(self.n_in, self.n_out)
        self.b = np.zeros(self.n_out)

    def get_shape(self):
        return self._shape

    def get_weights(self):
        return self.w, self.b

    def is_input_layer(self):
        return False

    def predict(self, tensor):
        h = tensor.dot(self.w) + self.b
        if self.activation:
            h = self.activation(h)
        return h

    def feedforward(self, tensor: np.array):
        y_pred = self.predict(tensor)
        return y_pred


class InputLayer(Layer):
    def is_input_layer(self):
        return True
