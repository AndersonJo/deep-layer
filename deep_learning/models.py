import numpy as np

from deep_learning.exceptions import LayerNotFound
from deep_learning.layers import Layer


class BaseModel(object):
    def __init__(self):
        self.layers = list()
        self.x_train: np.array = None
        self.y_train: np.array = None

    def compile(self, optimizer: str, loss: str):
        if not self.layers:
            raise LayerNotFound('Layer is not found. you must add at least one layer to the model')

        self.layers[0].is_input_layer = True

        prev_layer: Layer = None
        for layer in self.layers:
            layer.compile(prev_layer)
            prev_layer = layer

    def predict(self, x):
        tensor = x
        for layer in self.layers:
            tensor = layer.feedforward(tensor)
        return tensor

    def feedforward(self, x):
        tensors = []
        tensor = x
        for layer in self.layers:
            tensor = layer.feedforward(tensor)
            tensors.append(tensor)
        return tensors


class Model(BaseModel):
    def __init__(self):
        super(Model, self).__init__()

    def add(self, layer: Layer):
        self.layers.append(layer)

    def compile(self, optimizer=None, loss=None):
        super(Model, self).compile(optimizer, loss)

    def fit(self,
            x_train: np.array,
            y_train: np.array,
            shuffle: bool = True,
            batch_size: int = 35):

        x_train: np.array = np.array(x_train)
        y_train: np.array = np.array(y_train)
        N = len(x_train)

        print(x_train.shape)
        for idx in range(0, batch_size, batch_size):
            sample_x = x_train[idx: idx + batch_size]
            sample_y = y_train[idx: idx + batch_size]

            # Feedforward
            self.feedforward(sample_x)

            # Backpropagation
            tensors.reverse()
            tensor = tensors[0]
            tensors = tensors[1:]
            tensors.append(x_train)
            for prev_tensor, layer in zip(tensors, self.layers[::-1]):
                layer.backpropagation(prev_tensor, tensor)
                tensor = prev_tensor
