import numpy as np
from itertools import cycle
from random import shuffle as shuffle_list

class RollingSequence:
    """Helper class for creating batches for rolling sequence.

    Create batches of size `batch_size` that contain indices in `range(data_size)`.
    To that end, the data indices are repeated (rolling), either in ascending order or
    shuffled if `shuffle=True`. If taking batches sequentially, all data indices will
    appear equally often. If the size of the dataset isn't a multiple of the batch size,
    then successive sets of batch indices will roll over. For example, if batch size is
    4 and indices span 0-9, then the batches will be (0-3), (4-7), (8-9 and 0-1), (2-5),
    and so forth.
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
        self.indices = list(range(self.data_size))
        if self.shuffle:
            shuffle_list(self.indices)
        self.cycler = cycle(self.indices)

    def __len__(self):
        return self.length

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def batch(self, ):
        return [next(self.cycler) for _ in range(self.batch_size)]

    def __getitem__(self, i):
        return self.batch()