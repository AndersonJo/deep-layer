import numpy as np
import math


class Sample(object):
    def __init__(self, x: np.array, y: np.array = None, batch: int = 32):
        self.x: np.array = x.copy()
        self.y: np.array = y.copy() if y is not None else None

        self.batch: int = batch

        self._n = len(x)
        self._n_feature = x.shape[-1]

    def samples(self, n=10000, train=True, shuffle=True):
        split_n = len(self.x) - self.batch
        if split_n - self.batch <= 0:
            raise Exception('batch size is bigger than data size')

        if train:
            iter_range = range(n)
        else:
            split_n = int(math.ceil(len(self.x) / self.batch))
            iter_range = range(0, len(self.x), self.batch)

        for step in iter_range:
            i = step % split_n

            if i == 0 and shuffle:
                self.shuffle()

            sample_x = self.get_sample(self.x, idx=i)
            sample_y = self.get_sample(self.y, idx=i) if self.y is not None else None

            # Return
            if self.y is not None:
                yield sample_x, sample_y
            else:
                yield sample_x

    def get_sample(self, data: np.array, idx: int):

        sample = data[idx:idx + self.batch]
        n = len(sample)
        if n != self.batch and n != 0:
            sample = self.make_same(data, sample)
        return sample

    def make_same(self, data, sample):
        need_n = self.batch - len(sample)
        n = len(data)
        indices = np.random.randint(0, n, size=need_n)
        con_sample = np.concatenate((sample, data[indices]), axis=0)

        return con_sample

    def zeropad(self, data):
        if data is None:
            return None
        padded_data = np.zeros((self.batch, self._n_feature))
        padded_data[:len(data)] = data
        return padded_data

    def shuffle(self):
        rands = np.random.permutation(self._n)
        self.x = self.x[rands]
        if self.y is not None:
            self.y = self.y[rands]
