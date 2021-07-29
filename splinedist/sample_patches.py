from csbdeep.utils import _raise
import numpy as np


def get_valid_inds(datas, patch_size, patch_filter=None):
    len(patch_size) == datas[0].ndim or _raise(ValueError())

    if not all((a.shape == datas[0].shape for a in datas)):
        raise ValueError(
            "all input shapes must be the same: %s"
            % (" / ".join(str(a.shape) for a in datas))
        )

    if not all((0 < s <= d for s, d in zip(patch_size, datas[0].shape))):
        raise ValueError(
            "patch_size %s negative or larger than data shape %s along some dimensions"
            % (str(patch_size), str(datas[0].shape))
        )

    if patch_filter is None:
        patch_mask = np.ones(datas[0].shape, dtype=np.bool)
    else:
        patch_mask = patch_filter(datas[0], patch_size)

    # get the valid indices

    border_slices = tuple(
        [slice(p // 2, s - p + p // 2 + 1) for p, s in zip(patch_size, datas[0].shape)]
    )
    valid_inds = np.where(patch_mask[border_slices])
    valid_inds = tuple(v + s.start for s, v in zip(border_slices, valid_inds))
    return valid_inds
