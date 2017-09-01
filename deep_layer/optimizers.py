class BaseOptimizer(object):
    def __init__(self, lr: float = 0.001):
        """
        :param lr: Learning Rate
        """
        self.lr = lr


class StochasticGradientDescent(BaseOptimizer):
    pass


class Momentum(BaseOptimizer):
    def __init__(self, lr: float = 0.001, gamma: float = 0.5):
        super(Momentum, self).__init__(lr=lr)
        self.gamma = gamma

    def __call__(self, layer, delta_w, delta_b, n: int):
        layer.update_w = self.gamma * layer.update_w + - 2 / n * self.lr * delta_w
        layer.update_b = self.gamma * layer.update_b + - 2 / n * self.lr * delta_b
        return layer
