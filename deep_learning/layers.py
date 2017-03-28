import numpy as np

from deep_learning.activations import activations


class Layer(object):
    def __init__(self, n_out: int, activation: str = None, batch_input_shape: tuple = None,
                 name: str = None):
        self.name = name
        self.n_out: int = n_out
        self.activation = activations[activation] if activation else None
        self.dactivation = activations[f'd{activation}'] if activation else None
        self.batch_input_shape: tuple = batch_input_shape
        self.is_input_layer = False

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

    def predict(self, tensor):
        h = tensor.dot(self.w) + self.b
        if self.activation:
            h = self.activation(h)
        return h

    def feedforward(self, tensor: np.array):
        y_pred = self.predict(tensor)
        return y_pred

    def backpropagation(self, prev_tensor, tensor):
        pass

    def __str__(self):
        return f'<Layer {self.name}>'
