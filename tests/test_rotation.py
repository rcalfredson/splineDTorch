import numpy as np
from splinedist.rotation import sample_patches


def test_sample_patches_force_even():
    # Fake mask + image (odd-sized)
    mask = np.ones((15, 21), dtype=np.uint8)
    img = np.ones((15, 21, 3), dtype=np.uint8) * 127
    data = (np.stack([mask]), img)

    # Call sample_patches with bypass=True so it doesn't try real rotation
    res = sample_patches(data, patch_size=(15, 21), bypass=True, force_even=True)

    # res is a list of arrays, each (1, H, W)
    for arr in res:
        print('arr shape:', arr.shape)
        assert arr.shape[1] % 2 == 0
        assert arr.shape[2] % 2 == 0
