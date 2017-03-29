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

        prev_layer: Layer = None
        for layer in self.layers:
            layer.compile(prev_layer)
            prev_layer = layer

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

        # Prepare for Backpropagation
        outputs.insert(0, x)

        delta = None
        for i in range(len(self.layers) - 1, 0, -1):
            output: np.array = outputs[i + 1]
            prev_output: np.array = outputs[i]
            layer: Layer = self.layers[i]
            prev_layer: Layer = self.layers[i - 1]

            if layer.is_output_layer():
                # 2/N * (y_true - y_pred)
                loss = self.dloss(y, output)  # (35, 1)

                if layer.dactivation:
                    # TODO
                    pass
                else:
                    delta = loss * prev_output
            else:
                w, b = layer.get_weights()
                print(w.shape, b.shape)

                # print(delta.shape)

                # for i, (prev_output, layer) in enumerate(zip(outputs, self.layers[::-1])):
                #     if i == 0:
                #         # 2/N * (y_true - y_pred)
                #         l1 = self.dloss(y, output)
                #         if layer.dactivation:
                #             # TODO
                #             delta *= prev_output.T.dot(layer.dactivation(output))
                #         else:
                #             delta = l1 * prev_output
                #
                #     else:
                #         w, b = layer.get_weights()
                #         # delta^{(l+1)} dot weight^{(l)}
                #         delta = delta.dot(w.T)
                #         if layer.dactivation:
                #             # print(delta.T.shape, prev_tensor.shape)
                #             # print(layer.dactivation(tensor).shape)
                #             # print(layer.dactivation(tensor) * prev_tensor)
                #             print(layer.dactivation(output).shape, delta.shape)
                #             # delta * prev_output.T.dot(layer.dactivation(output))
                #             break
                #
                #             # delta.dot(w) * layer.dactivation(tensor) * prev_tensor
                #             # delta = delta.T * layer.dactivation(tensor)
                #
                #     print('delta:', delta.shape)
                #     output = prev_output
                #     print(i)

    def shuffle(self, x: np.array, y: np.array):
        N = len(x)
        rands = np.random.permutation(N)
        x = x[rands]
        y = y[rands]
        return x, y

    def get_batch_samples(self, x, y, batch_size, step=None):
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
                sample_x, sample_y = self.get_batch_samples(x_train, y_train, batch_size, step)

                # Feedforward
                tensors = self.feedforward(sample_x)

                # Backpropagation
                self.backpropagation(tensors, sample_x, sample_y, eta=learning_rate)
                break
