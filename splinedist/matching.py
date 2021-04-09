from csbdeep.utils import _raise
import numpy as np


def label_are_sequential(y):
    """ returns true if y has only sequential labels from 1... """
    labels = np.unique(y)
    return (set(labels) - {0}) == set(range(1, 1 + labels.max()))


def is_array_of_integers(y):
    return isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.integer)


def check_label_array(y, name=None, check_sequential=False):
    err = ValueError(
        "{label} must be an array of {integers}.".format(
            label="labels" if name is None else name,
            integers=("sequential " if check_sequential else "")
            + "non-negative integers",
        )
    )
    is_array_of_integers(y) or _raise(err)
    if check_sequential:
        label_are_sequential(y) or _raise(err)
    else:
        y.min() >= 0 or _raise(err)
    return True