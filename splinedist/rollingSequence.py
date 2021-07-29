import numpy as np
from collections import Counter

class RollingSequence:
    """Helper class for creating batches for rolling sequence.

    Create batches of size `batch_size` that contain indices in `range(data_size)`.
    To that end, the data indices are repeated (rolling), either in ascending order or
    shuffled if `shuffle=True`. If taking batches sequentially, all data indices will
    appear equally often. All calls to `batch(i)` will return the same batch for same i.
    Parameter `length` will only determine the result of `len`, it has no effect otherwise.
    Note that batch_size is allowed to be larger than data_size.
    """

    def __init__(self, data_size, batch_size, length=None, shuffle=True, rng=None):
        if rng is None:
            rng = np.random
        self.data_size = int(data_size)
        self.batch_size = int(batch_size)
        self.length = (
            2 ** 63 - 1 if length is None else int(length)
        )  # 2**63-1 is max possible value
        self.shuffle = bool(shuffle)
        self.index_gen = rng.permutation if self.shuffle else np.arange
        self.index_map = {}
        # self.indices_returned = []

    def __len__(self):
        return self.length

    def _index(self, loop):
        if loop in self.index_map:
            return self.index_map[loop]
        else:
            return self.index_map.setdefault(loop, self.index_gen(self.data_size))

    def reset_index_map(self):
        self.index_map = {}
        # self.indices_returned = []

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def batch(self, i):
        pos = i * self.batch_size
        loop = pos // self.data_size
        pos_loop = pos % self.data_size
        sl = slice(pos_loop, pos_loop + self.batch_size)
        index = self._index(loop)
        _loop = loop
        while sl.stop > len(index):
            _loop += 1
            index = np.concatenate((index, self._index(_loop)))
        # print(f"### - batch({i:02}) -> {tuple(index[sl])}", flush=True)
        # self.indices_returned += list(index[sl])
        return index[sl]

    def __getitem__(self, i):
        return self.batch(i)