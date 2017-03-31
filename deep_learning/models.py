import numpy as np

from deep_learning.exceptions import LayerNotFound
from deep_learning.layers import Layer, BaseLayer
from deep_learning.costs import losses, dmean_squared_error
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

    def backpropagation(self,
                        outputs: np.array,
                        x: np.array,
                        y: np.array,
                        n_data: int = 2,
                        eta: float = 0.01):
        outputs.insert(0, x)

        N = len(self.layers)
        deltas = []
        delta = None

        loss = np.nan
        for i in range(N, 0, -1):
            output: np.array = outputs[i]
            prev_output: np.array = outputs[i - 1]
            layer: Layer = self.layers[i - 1]

            if i == N:
                d1 = self.dloss(y, output)
                loss = np.sum(d1)
            else:
                layer2: Layer = self.layers[i]
                w2, b2 = layer2.get_weights()
                d1 = w2.dot(delta)

            d2 = layer.dactivation(output).T
            delta = d1 * d2

            delta_w = delta.dot(prev_output).T
            delta_b = delta.reshape([-1])
            deltas.append((delta_w, delta_b))

        layers = self.layers[::-1]
        for i in range(len(deltas) - 1, -1, -1):
            delta_w, delta_b = deltas[i]
            layer = layers[i]

            layer.update_w = 0.5 * layer.update_w + - 2 / n_data * eta * delta_w
            layer.update_b = 0.5 * layer.update_b + - 2 / n_data * eta * delta_b

            layer.w += layer.update_w
            layer.b += layer.update_b

        return dict(loss=loss)

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

            losses = list()
            for step in range(0, N):
                sample_x, sample_y = self.get_batch_samples(x_train, y_train, 1, step)

                # Feedforward
                tensors = self.feedforward(sample_x)

                # Backpropagation
                result = self.backpropagation(tensors, sample_x, sample_y, n_data=N, eta=learning_rate)
                losses.append(result['loss'])

            print('epoch:', epoch, 'loss:', np.sum(losses))
