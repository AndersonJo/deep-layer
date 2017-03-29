import numpy as np

from deep_learning.exceptions import LayerNotFound
from deep_learning.layers import Layer, BaseLayer
from deep_learning.loses import losses, dmean_squared_error
from deep_learning.utils import _get_function


class BaseModel(object):
    def __init__(self):
        self.layers = list()
        self.x_train: np.array = None
        self.y_train: np.array = None

        self.optimizer = None
        self.loss = None
        self.dloss = None

    def compile(self, optimizer: str, loss: str):
        if not self.layers:
            raise LayerNotFound('Layer is not found. you must add at least one layer to the model')

        self.loss = _get_function(losses, loss, loss)
        self.dloss = _get_function(losses, f'd{loss}', loss)

        for layer in self.layers:
            layer.compile()

        layer._is_output_layer = True

    def predict(self, x):
        tensor = x
        for layer in self.layers:
            tensor = layer.feedforward(tensor)
        return tensor

    def feedforward(self, x: np.array):
        outputs = []
        tensor = x
        for layer in self.layers:
            tensor = layer.feedforward(tensor)
            outputs.append(tensor)
        return outputs

    def backpropagation(self, outputs: np.array, x: np.array, y: np.array, eta: float = 0.01):
        outputs.insert(0, x)
        N = len(self.layers)
        deltas = []
        delta = None
        for i in range(N, 0, -1):
            output: np.array = outputs[i]
            prev_output: np.array = outputs[i - 1]
            layer: Layer = self.layers[i - 1]

            # prev_layer: Layer = self.layers[i - 2]
            if i == N:
                # 2/N * (y_true - y_pred)
                d1 = self.dloss(y, output)  # (35, 1)
            else:
                layer2: Layer = self.layers[i]
                w2, b2 = layer2.get_weights()
                d1 = w2.dot(delta)

            d2 = layer.dactivation(output).T
            delta = d1 * d2

            delta_w = delta.dot(prev_output).T
            delta_b = delta.reshape(-1)
            deltas.append((delta_w, delta_b))

        layers = self.layers[::-1]
        for i in range(len(deltas) - 1, -1, -1):
            delta_w, delta_b = deltas[i]
            layer = layers[i]

            update_w = - eta * delta_w
            update_b = - eta * delta_b

            layer.w += update_w
            layer.b += update_b

            # print(delta_w.shape, delta_b.shape, layer, layer.w.shape, layer.b.shape, update_w.shape, update_b.shape)

    @staticmethod
    def shuffle(x: np.array, y: np.array):
        N = len(x)
        rands = np.random.permutation(N)
        x = x[rands]
        y = y[rands]
        return x, y

    @staticmethod
    def get_batch_samples(x, y, batch_size, step=None):
        if not step:
            step = np.random.randint(len(x) - batch_size)

        sample_x = x[step: step + batch_size]
        sample_y = y[step: step + batch_size]
        return sample_x, sample_y


class Model(BaseModel):
    def __init__(self):
        super(Model, self).__init__()

    def add(self, layer: BaseLayer):
        self.layers.append(layer)

    def compile(self, optimizer=None, loss=None):
        super(Model, self).compile(optimizer, loss)

    def fit(self,
            x_train: np.array,
            y_train: np.array,
            learning_rate=0.01,
            shuffle: bool = True,
            batch_size: int = 35,
            epochs=10):
        x_train: np.array = np.array(x_train)
        y_train: np.array = np.array(y_train)
        N = len(x_train)

        for epoch in range(epochs):
            if shuffle:
                x_train, y_train = self.shuffle(x_train, y_train)

            for step in range(0, N):
                # sample_x, sample_y = self.get_batch_samples(x_train, y_train, batch_size, step)
                sample_x = np.atleast_2d(x_train[step])
                sample_y = np.atleast_2d(y_train[step])
                # sample_x = x_train[step]
                # sample_y = y_train[step]

                # Feedforward
                tensors = self.feedforward(sample_x)

                # Backpropagation
                self.backpropagation(tensors, sample_x, sample_y, eta=learning_rate)
                break
