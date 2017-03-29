import numpy as np

from deep_learning.activations import activations
from deep_learning.utils import _get_function


class BaseLayer(object):
    def __init__(self,
                 n_out: int,
                 name: str = None,
                 batch_input_shape: tuple = None):
        self.n_out: int = n_out
        self.name = name
        self.batch_input_shape: tuple = batch_input_shape

        self._is_output_layer = False

    def compile(self, prev_layer):
        raise NotImplementedError('compile method should be implemented')

    def is_input_layer(self):
        raise NotImplementedError('is_input_layer method should be implemented')

    def is_output_layer(self):
        return self._is_output_layer

    def feedforward(self, tensor: np.array):
        raise NotImplementedError('feedforward method should be implemented')

    def __str__(self):
        return f'<Layer {self.name}>'


class InputLayer(BaseLayer):
    def __init__(self,
                 name: str = None,
                 batch_input_shape: tuple = None):
        super(InputLayer, self).__init__(batch_input_shape[-1],
                                         batch_input_shape=batch_input_shape,
                                         name=name)

    def compile(self, prev_layer):
        pass

    def is_input_layer(self):
        return True

    def feedforward(self, tensor: np.array):
        return tensor


class Layer(BaseLayer):
    def __init__(self, n_out: int,
                 activation: str = None, dactivation=None,
                 batch_input_shape: tuple = None,
                 name: str = None):
        super(Layer, self).__init__(n_out,
                                    name=name,
                                    batch_input_shape=batch_input_shape)

        self.activation = _get_function(activations, activation, activation)
        self.dactivation = _get_function(activations, f'd{activation}', dactivation)
        self.w: np.array = None
        self.b: np.array = None

        # Shape
        self._shape = None

    def compile(self, prev_layer):
        self._create_weight(prev_layer)

    def _create_weight(self, prev_layer) -> None:
        w_shape, b_shape = self._get_shape(prev_layer)
        self.w = np.random.randn(*w_shape)
        self.b = np.zeros(b_shape)

    def _get_shape(self, prev_layer) -> (tuple, int):
        # Input Layer
        if not prev_layer:
            input_shape = list(self.batch_input_shape)
            input_shape.remove(None)
            input_shape.append(self.n_out)
        else:
            input_shape = [prev_layer.n_out, self.n_out]

        return tuple(input_shape), self.n_out

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

        # def backpropagation(self, prev_tensor, tensor):
        #     dactivation = np.array([1])
        #     if self.dactivation:
        #         dactivation = self.dactivation(tensor)
        #     return dactivation.T * prev_tensor
