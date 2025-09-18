import numpy as np
from splinedist.utils import pad_to_even


def test_pad_to_even_preserves_even_shape():
    arr = np.zeros((16, 20))
    padded = pad_to_even(arr)
    assert padded.shape == (16, 20)  # no change
    assert np.allclose(padded, arr)  # values preserved


def test_pad_to_even_pads_odd_height():
    arr = np.ones((15, 20))  # odd height
    padded = pad_to_even(arr)
    assert padded.shape == (16, 20)  # height padded
    assert np.all(padded[:-1, :] == 1)
    assert np.all(padded[-1, :] == 0)  # new row is zeros


def test_pad_to_even_pads_odd_width():
    arr = np.ones((16, 21))
    padded = pad_to_even(arr)
    assert padded.shape == (16, 22)
    assert np.all(padded[:, :-1] == 1)
    assert np.all(padded[:, -1] == 0)


def test_pad_to_even_pads_both_dims_and_preserves_channels():
    arr = np.ones((15, 21, 3))
    padded = pad_to_even(arr)
    assert padded.shape == (16, 22, 3)

    # original values preserved
    assert np.all(padded[:15, :21, :] == 1)

    # padded parts are zeros
    assert np.all(padded[15, :, :] == 0)
    assert np.all(padded[:, 21, :] == 0)


def test_pad_to_even_handles_leading_singleton():
    arr = np.ones((1, 15, 21))
    padded = pad_to_even(arr)
    assert padded.shape == (1, 16, 22)

    # original content preserved
    assert np.all(padded[0, :15, :21] == 1)
    assert np.all(padded[0, 15, :] == 0)
    assert np.all(padded[0, :, 21] == 0)
