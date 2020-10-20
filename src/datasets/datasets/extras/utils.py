import random
from typing import List
import math
import numpy as np


def create_splits(data: List,
                  train: float = 0.8,
                  dev: float = 0.1,
                  test: float = 0.1,
                  shuffle: bool = True):

    if not np.allclose(train + dev + test, 1.0):
        raise ValueError(
            "test+dev+train should be 1.0 but is {}".format(train + dev +
                                                            test))
    num_samples = len(data)
    num_train = int(math.floor(num_samples * train))
    num_val = int(math.floor(num_samples * dev))
    num_test = num_samples - num_train - num_val

    if shuffle:
        random.shuffle(data)
    train = [data.pop() for i in range(num_train)]
    dev = [data.pop() for i in range(num_val)]
    test = [data.pop() for i in range(num_test)]

    if len(data) > 0:
        raise RuntimeError

    return train, dev, test
