class BaseModel(object):
    def __init__(self):
        self.layers = list()
        self.x_train = None
        self.y_train = None

    def layer_shape(self):
        n_layer = len(self.layers)

        for i, layer in enumerate(self.layers):
            if i == 0:
                print(layer.batch_input_shape)


class Model(BaseModel):
    def __init__(self):
        super(Model, self).__init__()

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer=None, loss=None, batch_size=30):
        self.layer_shape()

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
