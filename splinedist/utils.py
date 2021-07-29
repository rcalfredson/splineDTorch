from datetime import datetime
from csbdeep.utils import _raise
import cv2
import numpy as np
import os
import platform
from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes
from scipy.ndimage.measurements import find_objects
from skimage.measure import regionprops
import spline_generator as sg
from splinedist.constants import DEVICE
import torch
from typing import Union
import warnings

has_cv2_v4 = cv2.__version__.startswith("4")
onload_ts = datetime.now()


def _is_power_of_2(i):
    assert i > 0
    e = np.log2(i)
    return e == int(e)


def _edt_dist_func(anisotropy):
    try:
        from edt import edt as edt_func

        # raise ImportError()
        dist_func = lambda img: edt_func(
            np.ascontiguousarray(img > 0), anisotropy=anisotropy
        )
    except ImportError:
        dist_func = lambda img: distance_transform_edt(img, sampling=anisotropy)
    return dist_func


def edt_prob(lbl_img, anisotropy=None):
    """Perform EDT on each labeled object and normalize."""

    def grow(sl, interior):
        return tuple(
            slice(s.start - int(w[0]), s.stop + int(w[1])) for s, w in zip(sl, interior)
        )

    def shrink(interior):
        return tuple(slice(int(w[0]), (-1 if w[1] else None)) for w in interior)

    constant_img = lbl_img.min() == lbl_img.max() and lbl_img.flat[0] > 0
    if constant_img:
        lbl_img = np.pad(lbl_img, ((1, 1),) * lbl_img.ndim, mode="constant")
        warnings.warn(
            "EDT of constant label image is ill-defined. (Assuming background around it.)"
        )
    dist_func = _edt_dist_func(anisotropy)
    objects = find_objects(lbl_img)
    prob = np.zeros(lbl_img.shape, np.float32)
    for i, sl in enumerate(objects, 1):
        # i: object label id, sl: slices of object in lbl_img
        if sl is None:
            continue
        interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, lbl_img.shape)]
        # 1. grow object slice by 1 for all interior object bounding boxes
        # 2. perform (correct) EDT for object with label id i
        # 3. extract EDT for object of original slice and normalize
        # 4. store edt for object only for pixels of given label id i
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl, interior)] == i
        mask = grown_mask[shrink_slice]
        edt = dist_func(grown_mask)[shrink_slice][mask]
        prob[sl][mask] = edt / (np.max(edt) + 1e-10)
    if constant_img:
        prob = prob[(slice(1, -1),) * lbl_img.ndim].copy()
    return prob


def calculate_extents(lbl, func=np.median):
    """ Aggregate bounding box sizes of objects in label images. """
    if isinstance(lbl, (tuple, list)) or (
        isinstance(lbl, np.ndarray) and lbl.ndim == 4
    ):
        return func(
            np.stack([calculate_extents(_lbl, func) for _lbl in lbl], axis=0), axis=0
        )

    n = lbl.ndim
    n in (2, 3) or _raise(
        ValueError(
            "label image should be 2- or 3-dimensional (or pass a list of these)"
        )
    )

    regs = regionprops(lbl)
    if len(regs) == 0:
        return np.zeros(n)
    else:
        extents = np.array([np.array(r.bbox[n:]) - np.array(r.bbox[:n]) for r in regs])
        return func(extents, axis=0)


def wrapIndex(t, k, M, half_support):
    wrappedT = t - k
    t_left = t - half_support
    t_right = t + half_support
    if k < t_left:
        if t_left <= k + M <= t_right:
            wrappedT = t - (k + M)
    elif k > t + half_support:
        if t_left <= k - M <= t_right:
            wrappedT = t - (k - M)
    return wrappedT




def data_dir(must_exist=True):
    prospective_dir = os.path.join("data_by_host", f"{platform.node()}_{onload_ts}".replace(":", '-'))
    if (must_exist and os.path.isdir(prospective_dir)) or not must_exist:
        return prospective_dir
    elif must_exist and not os.path.isdir(prospective_dir):
        return './'


def phi_generator(M, contoursize_max, debug=False):
    ts = np.linspace(0, float(M), num=contoursize_max, endpoint=False)
    wrapped_indices = np.array([[wrapIndex(t, k, M, 2) for k in range(M)] for t in ts])
    vfunc = np.vectorize(sg.B3().value)
    phi = vfunc(wrapped_indices)
    phi = phi.astype(np.float32)
    np.save(
        os.path.join("debug" if debug else data_dir(), "phi_" + str(M) + ".npy"), phi
    )
    return


def grid_generator(M, patch_size, grid_subsampled):
    coord = np.ones((patch_size[0], patch_size[1], M, 2))

    xgrid_points = np.linspace(0, coord.shape[0] - 1, coord.shape[0])
    ygrid_points = np.linspace(0, coord.shape[1] - 1, coord.shape[1])
    xgrid, ygrid = np.meshgrid(xgrid_points, ygrid_points)
    xgrid, ygrid = np.transpose(xgrid), np.transpose(ygrid)
    grid = np.stack((xgrid, ygrid), axis=2)
    grid = np.expand_dims(grid, axis=2)
    grid = np.repeat(grid, coord.shape[2], axis=2)
    grid = np.expand_dims(grid, axis=0)

    grid = grid[:, 0 :: grid_subsampled[0], 0 :: grid_subsampled[1]]
    grid = grid.astype(np.float32)
    np.save(os.path.join(data_dir(), "grid_" + str(M) + ".npy"), grid)
    return


def normalize_grid(grid, n):
    try:
        grid = tuple(grid)
        (
            len(grid) == n
            and all(map(np.isscalar, grid))
            and all(map(_is_power_of_2, grid))
        ) or _raise(TypeError())
        return tuple(int(g) for g in grid)
    except (TypeError, AssertionError):
        raise ValueError(
            "grid = {grid} must be a list/tuple of length {n} with values that are power of 2".format(
                grid=grid, n=n
            )
        )


def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(0.01 * float(q) * (t.numel() - 1))
    result = t.reshape(-1).kthvalue(k).values.item()
    return result


def normalize(x, pmin=3, pmax=99.8, ind_norm=True, clip=False, eps=1e-20):
    """Percentile-based image normalization."""
    if not ind_norm:
        raise NotImplementedError
    try:
        mi = torch.stack(
            [torch.quantile(x[:, :, i], pmin / 100) for i in range(x.shape[-1])]
        )
        ma = torch.stack(
            [torch.quantile(x[:, :, i], pmax / 100) for i in range(x.shape[-1])]
        )

    except AttributeError:
        mi = torch.tensor([percentile(x[:, :, i], pmin) for i in range(x.shape[-1])]).float().to(DEVICE)
        ma = torch.tensor([percentile(x[:, :, i], pmax) for i in range(x.shape[-1])]).float().to(DEVICE)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20):
    try:
        import numexpr

        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

    if clip:
        x = torch.clamp(x, 0, 1)

    return x


def fill_label_holes(lbl_img, **kwargs):
    """Fill small holes in label image."""

    def grow(sl, interior):
        return tuple(
            slice(s.start - int(w[0]), s.stop + int(w[1])) for s, w in zip(sl, interior)
        )

    def shrink(interior):
        return tuple(slice(int(w[0]), (-1 if w[1] else None)) for w in interior)

    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue
        interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl, interior)] == i
        mask_filled = binary_fill_holes(grown_mask, **kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled


def get_contoursize_max(Y_trn):
    contoursize = []
    for i in range(len(Y_trn)):
        mask = Y_trn[i]
        obj_list = np.unique(mask)
        obj_list = obj_list[1:]

        for j in range(len(obj_list)):
            mask_temp = mask.copy()
            mask_temp[mask_temp != obj_list[j]] = 0
            mask_temp[mask_temp > 0] = 1

            mask_temp = mask_temp.astype(np.uint8)
            if has_cv2_v4:
                contours, _ = cv2.findContours(
                    mask_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
                )
            else:
                _, contours, _ = cv2.findContours(
                    mask_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
                )
            areas = [cv2.contourArea(cnt) for cnt in contours]
            max_ind = np.argmax(areas)
            contour = np.squeeze(contours[max_ind])
            contour = np.reshape(contour, (-1, 2))
            contour = np.append(contour, contour[0].reshape((-1, 2)), axis=0)
            contoursize = np.append(contoursize, contour.shape[0])

    contoursize_max = np.amax(contoursize)
    return contoursize_max
