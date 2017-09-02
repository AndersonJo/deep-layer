import numpy as np

from deep_layer.exceptions import LayerNotFound
from deep_layer.layers import Layer, BaseLayer
from deep_layer.costs import losses
from deep_layer.sample import Sample
from deep_layer.utils import _get_function


class BaseModel(object):
    def __init__(self):
        self.layers = list()
        self.x_train: np.array = None
        self.y_train: np.array = None

        self.batch_size = None
        self.optimizer = None
        self.loss = None
        self.dloss = None

        self.last_n_in = 0
        self.last_n_out = 0

    def compile(self, optimizer, loss: str, batch=32):
        if not self.layers:
            raise LayerNotFound('Layer is not found. you must add at least one layer to the model')

        self.batch_size = batch
        self.optimizer = optimizer
        self.loss = _get_function(losses, loss, loss)
        self.dloss = _get_function(losses, f'd{loss}', loss)

        for layer in self.layers:
            layer.compile(batch=batch)

        last_layer = self.layers[-1]
        self.last_n_in: int = last_layer.n_in
        self.last_n_out: int = last_layer.n_out

    def predict(self, x):
        n = len(x)
        sample = Sample(x, batch=self.batch_size)

        response = list()
        for tensor in sample.samples(train=False, shuffle=False):
            for layer in self.layers:
                tensor = layer.feedforward(tensor)
            response.append(tensor)

        response = np.array(response)
        response = response.reshape(-1, self.last_n_out)[:n]
        print('response:', response.shape)
        return response

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
                        n_data: int = 2):
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
                d1 = delta.dot(w2.T)

            delta = d1 * layer.dactivation(output)
            delta_w = prev_output.T.dot(delta)
            delta_b = delta
            deltas.append((delta_w, delta_b))

        layers = self.layers[::-1]
        for i in range(len(deltas) - 1, -1, -1):
            delta_w, delta_b = deltas[i]
            layer = layers[i]

            layer.zero_grad()
            layer = self.optimizer(layer, delta_w, delta_b, n_data)
            layer.update()

        return dict(loss=loss)


class Model(BaseModel):
    def __init__(self):
        super(Model, self).__init__()

    def add(self, layer: BaseLayer):
        self.layers.append(layer)

    def fit(self,
            x_train: np.array,
            y_train: np.array,
            shuffle: bool = True,
            epochs=10):

        sample = Sample(x_train, y_train, batch=self.batch_size)
        N = len(x_train)

        for epoch in range(epochs):
            sample.shuffle()

            losses = list()
            for i, (sample_x, sample_y) in enumerate(sample.samples(n=30000, shuffle=True)):
                # Feedforward
                tensors = self.feedforward(sample_x)

                # Backpropagation
                result = self.backpropagation(tensors, sample_x, sample_y, n_data=N)
                losses.append(result['loss'])

            print(f'[Epoch {epoch}] loss: {np.mean(losses)}')
