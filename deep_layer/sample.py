import numpy as np
import math


class Sample(object):
    def __init__(self, x: np.array, y: np.array = None, batch: int = 32):
        self.x: np.array = x.copy()
        self.y: np.array = y.copy() if y is not None else None
        self.batch: int = batch

        self._n = len(x)
        self._n_feature = x.shape[-1]

    def samples(self, n=0):
        N = max(int(math.ceil(self._n / self.batch)), n)
        for step in range(N):
            i = int(step / self._n)
            sample_x = self.x[i * self.batch:i * self.batch + self.batch]
            if self.y is not None:
                sample_y = self.y[i * self.batch:i * self.batch + self.batch]
                yield sample_x, sample_y
            else:
                yield sample_x

    def shuffle(self):
        rands = np.random.permutation(self._n)
        self.x = self.x[rands]
        self.y = self.y[rands]
