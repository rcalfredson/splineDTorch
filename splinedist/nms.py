import os
from splinedist.constants import DEVICE
import numpy as np
from splinedist.utils import data_dir, normalize_grid
import spline_generator as sg
from time import time
import torch


def non_maximum_suppression(
    coord,
    prob,
    grid=(1, 1),
    b=2,
    nms_thresh=0.5,
    prob_thresh=0.5,
    verbose=False,
    max_bbox_search=True,
):
    """2D coordinates of the polys that survive from a given prediction (prob, coord)

    prob.shape = (Ny,Nx)
    coord.shape = (Ny,Nx,2,n_params)

    b: don't use pixel closer than b pixels to the image boundary
    """
    global coordEval
    from stardist.lib.stardist2d import c_non_max_suppression_inds_old

    # TODO: using b>0 with grid>1 can suppress small/cropped objects at the image boundary

    assert prob.ndim == 2
    assert coord.ndim == 4
    grid = normalize_grid(grid, 2)

    mask = prob > prob_thresh
    if b is not None and b > 0:
        _mask = np.zeros_like(mask)
        _mask[b:-b, b:-b] = True
        mask &= _mask

    coord = np.transpose(coord, (0, 1, 3, 2))
    M = np.shape(coord)[2]

    phi = np.load(os.path.join(data_dir(), "phi_" + str(M) + ".npy"))
    # phi = torch.from_numpy(phi).float().to(DEVICE)
    SplineContour = sg.SplineCurveVectorized(
        M, sg.B3(), True, coord, useTorch=False
    )
    coord = SplineContour.sampleSequential(phi)
    # print('session eval time:', timeit.default_timer() - start_time)

    coord = np.transpose(coord, (0, 1, 3, 2))

    polygons = coord[mask]
    scores = prob[mask]

    # sort scores descendingly
    ind = np.argsort(scores)[::-1]
    survivors = np.zeros(len(ind), np.bool)
    polygons = polygons[ind]
    scores = scores[ind]

    if max_bbox_search:
        # map pixel indices to ids of sorted polygons (-1 => polygon at that pixel not a candidate)
        mapping = -np.ones(mask.shape, np.int32)
        mapping.flat[np.flatnonzero(mask)[ind]] = range(len(ind))
    else:
        mapping = np.empty((0, 0), np.int32)

    if verbose:
        t = time()

    survivors[ind] = c_non_max_suppression_inds_old(
        polygons.astype(np.int32),
        mapping,
        np.float32(nms_thresh),
        np.int32(max_bbox_search),
        np.int32(grid[0]),
        np.int32(grid[1]),
        np.int32(verbose),
    )

    if verbose:
        print("keeping %s/%s polygons" % (np.count_nonzero(survivors), len(polygons)))
        print("NMS took %.4f s" % (time() - t))

    points = np.stack([ii[survivors] for ii in np.nonzero(mask)], axis=-1)
    return points