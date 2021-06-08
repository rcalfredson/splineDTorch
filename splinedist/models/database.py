import timeit
from splinedist.rollingSequence import RollingSequence
from csbdeep.utils import _raise
import numpy as np
from skimage.segmentation import clear_border
from splinedist.constants import DEVICE
from splinedist.geometry.geom2d import spline_dist
from splinedist.rotation import sample_patches as sample_patches_rot
from splinedist.sample_patches import get_valid_inds
from splinedist.utils import edt_prob
import torch
import sys

import matplotlib.pyplot as plt
import cProfile


class SplineDistDataBase(RollingSequence):
    def __init__(
        self,
        X,
        Y,
        n_params,
        grid,
        batch_size,
        patch_size,
        length,
        use_gpu=False,
        sample_ind_cache=True,
        maxfilter_patch_size=None,
        augmenter=None,
        foreground_prob=0,
    ):
        super().__init__(
            data_size=len(X), batch_size=batch_size, length=length, shuffle=True
        )

        if isinstance(X, (np.ndarray, tuple, list)):
            X = [x.astype(np.float32, copy=False) for x in X]

        # Y = [y.astype(np.uint16,  copy=False) for y in Y]

        # sanity checks
        assert len(X) == len(Y) and len(X) > 0
        nD = len(patch_size)
        assert nD in (2, 3)
        x_ndim = X[0].ndim
        assert x_ndim in (nD, nD + 1)

        if isinstance(X, (np.ndarray, tuple, list)) and isinstance(
            Y, (np.ndarray, tuple, list)
        ):
            all(
                y.ndim == nD and x.ndim == x_ndim and x.shape[:nD] == y.shape
                for x, y in zip(X, Y)
            ) or _raise("images and masks should have corresponding shapes/dimensions")
            # REFACTORED
        #             all(x.shape[:nD]>=patch_size for x in X) or _raise("Some images are too small for given patch_size {patch_size}".format(patch_size=patch_size))

        if x_ndim == nD:
            self.n_channel = None
        else:
            self.n_channel = X[0].shape[-1]
            assert all(x.shape[-1] == self.n_channel for x in X)
        assert 0 <= foreground_prob <= 1

        self.X, self.Y = X, Y
        # self.batch_size = batch_size
        self.n_params = n_params
        self.patch_size = patch_size
        self.ss_grid = (slice(None),) + tuple(slice(0, None, g) for g in grid)
        self.use_gpu = bool(use_gpu)
        if augmenter is None:
            augmenter = lambda *args: args
        callable(augmenter) or _raise(ValueError("augmenter must be None or callable"))
        self.augmenter = augmenter
        self.foreground_prob = foreground_prob

        if self.use_gpu:
            from gputools import max_filter

            self.max_filter = lambda y, patch_size: max_filter(
                y.astype(np.float32), patch_size
            )
        else:
            from scipy.ndimage.filters import maximum_filter

            self.max_filter = lambda y, patch_size: maximum_filter(
                y, patch_size, mode="constant"
            )

        self.maxfilter_patch_size = (
            maxfilter_patch_size
            if maxfilter_patch_size is not None
            else self.patch_size
        )

        self.sample_ind_cache = sample_ind_cache
        self._ind_cache_fg = {}
        self._ind_cache_all = {}

    def get_valid_inds(self, k, foreground_prob=None):
        if foreground_prob is None:
            foreground_prob = self.foreground_prob
        foreground_only = np.random.uniform() < foreground_prob
        _ind_cache = self._ind_cache_fg if foreground_only else self._ind_cache_all
        if k in _ind_cache:
            inds = _ind_cache[k]
        else:
            patch_filter = (
                (lambda y, p: self.max_filter(y, self.maxfilter_patch_size) > 0)
                if foreground_only
                else None
            )
            inds = get_valid_inds(
                (self.Y[k],) + self.channels_as_tuple(self.X[k]),
                self.patch_size,
                patch_filter=patch_filter,
            )
            if self.sample_ind_cache:
                _ind_cache[k] = inds
        if foreground_only and len(inds[0]) == 0:
            # no foreground pixels available
            return self.get_valid_inds(k, foreground_prob=0)
        return inds

    def channels_as_tuple(self, x):
        if self.n_channel is None:
            return (x,)
        else:
            return tuple(x[..., i] for i in range(self.n_channel))


class SplineDistDataStatic(SplineDistDataBase):
    def __init__(
        self,
        X,
        Y,
        batch_size,
        n_params,
        length,
        contoursize_max,
        patch_size=(256, 256),
        b=32,
        grid=(1, 1),
        shape_completion=False,
        augmenter=None,
        foreground_prob=0,
        **kwargs,
    ):
        super().__init__(
            X=X,
            Y=Y,
            n_params=n_params,
            grid=grid,
            batch_size=batch_size,
            patch_size=patch_size,
            length=length,
            augmenter=augmenter,
            foreground_prob=foreground_prob,
            **kwargs,
        )
        self.batch_size = batch_size
        self.contoursize_max = contoursize_max
        self.shape_completion = bool(shape_completion)
        if self.shape_completion and b > 0:
            self.b = slice(b, -b), slice(b, -b)
        else:
            self.b = slice(None), slice(None)

        self.indices_already_checked = set()
        self.check_for_duplicates = False

    def __getitem__(self, i):
        idx = self.batch(i)
        if i % self.length == 0:
            self.indices_already_checked = set()
        if self.check_for_duplicates:
            for i in idx:
                if i not in self.indices_already_checked:
                    self.indices_already_checked.add(i)
                else:
                    print("Found a duplicate index in one validation step")
                    print("i is", i)
                    raise BaseException
        # set a subset of X and Y
        # whose overall structure matches the originals.
        # for X, does that mean using tuple?
        X = [self.X[i] for i in idx]
        Y = [self.Y[i] for i in idx]
        X = np.stack(X)
        if X.ndim == 3:  # input image has no channel axis
            X = np.expand_dims(X, -1)
        prob = np.stack([edt_prob(lbl[self.b]) for lbl in Y])

        if self.shape_completion:
            Y_cleared = [clear_border(lbl) for lbl in Y]
            dist = np.stack(
                [
                    spline_dist(lbl, self.contoursize_max)[self.b + (slice(None),)]
                    for lbl in Y_cleared
                ]
            )
            dist_mask = np.stack([edt_prob(lbl[self.b]) for lbl in Y_cleared])
        else:
            sd_results = []
            for lbl in Y:
                sd_results.append(spline_dist(lbl, self.contoursize_max))
            dist = np.stack(sd_results)
            dist_mask = prob

        prob = np.expand_dims(prob, -1)
        dist_mask = np.expand_dims(dist_mask, -1)

        # subsample wth given grid
        prob = prob[self.ss_grid]

        # append dist_mask to dist as additional channel
        dist_mask = dist_mask[self.ss_grid]
        dist = dist[self.ss_grid]
        dist = np.concatenate([dist, dist_mask], axis=-1)
        num_instances = [len(np.unique(el)) - 1 for el in Y]

        return [X], [prob, dist], num_instances


class SplineDistData2D(SplineDistDataBase):
    def __init__(
        self,
        X,
        Y,
        batch_size,
        n_params,
        length,
        contoursize_max,
        patch_size=(256, 256),
        b=32,
        grid=(1, 1),
        shape_completion=False,
        augmenter=None,
        skip_empties=False,
        foreground_prob=0,
        n_samples=1,
        skip_dist_prob_calc=False,
        **kwargs,
    ):

        super().__init__(
            X=X,
            Y=Y,
            n_params=n_params,
            grid=grid,
            batch_size=batch_size,
            patch_size=patch_size,
            length=length,
            augmenter=augmenter,
            foreground_prob=foreground_prob,
            **kwargs,
        )

        self.shape_completion = bool(shape_completion)
        if self.shape_completion and b > 0:
            self.b = slice(b, -b), slice(b, -b)
        else:
            self.b = slice(None), slice(None)

        self.sd_mode = "opencl" if self.use_gpu else "cpp"

        self.skip_empties = skip_empties
        self.contoursize_max = contoursize_max
        self.n_samples = n_samples
        self.skip_dist_prob_calc = skip_dist_prob_calc

    def __getitem__(self, i):
        # start_t = timeit.default_timer()
        idx = self.batch(i)
        # original sample_patches
        # arrays = [sample_patches((self.Y[k],) + self.channels_as_tuple(self.X[k]),
        #                          patch_size=self.patch_size, n_samples=1,
        #                          valid_inds=self.get_valid_inds(k)) for k in idx]

        # random rotation sample_patches

        self.arrays = [
            sample_patches_rot((self.Y[k], self.X[k]), patch_size=self.patch_size, skip_empties=self.skip_empties)
            for k in idx
        ]
        np.set_printoptions(threshold=sys.maxsize)
        with open('debug_mask.txt', 'w') as f:
            print('deformed mask:', self.arrays[0][0], file=f)
        # sample_patch_time = timeit.default_timer()
        # print(f"time spent sampling patches: {sample_patch_time - start_t:.3f}")
        # pr = cProfile.Profile()
        # pr.enable()
        # self.arrays = [
        #     sample_patches(
        #         (self.Y[k],) + self.channels_as_tuple(self.X[k]),
        #         patch_size=self.patch_size,
        #         n_samples=self.n_samples,
        #         valid_inds=self.get_valid_inds(k),
        #     )
        #     for k in idx
        # ]
        if self.n_channel is None:
            X, Y = list(
                zip(
                    *[
                        (
                            x[0][self.b],
                            y[0],
                        )
                        for y, x in self.arrays
                    ]
                )
            )
        else:
            images, masks = [], []
            for y, *x in self.arrays:
                for i in range(x[0].shape[0]):
                    images.append(np.stack([_x[i] for _x in x], axis=-1))
                    masks.append(y[i])

            X, Y = list(zip(*[(images[i], masks[i]) for i in range(len(images))]))

        X, Y = tuple(zip(*tuple(self.augmenter(_x, _y) for _x, _y in zip(X, Y))))

        X = np.stack(X)
        if X.ndim == 3:  # input image has no channel axis
            X = np.expand_dims(X, -1)


        if not self.skip_dist_prob_calc:
            prob = np.stack([edt_prob(lbl[self.b]) for lbl in Y])

            if self.shape_completion:
                Y_cleared = [clear_border(lbl) for lbl in Y]
                dist = np.stack(
                    [
                        spline_dist(lbl, self.contoursize_max)[self.b + (slice(None),)]
                        for lbl in Y_cleared
                    ]
                )
                dist_mask = np.stack([edt_prob(lbl[self.b]) for lbl in Y_cleared])
            else:
                sd_results = []
                for lbl in Y:
                    sd_results.append(spline_dist(lbl, self.contoursize_max))
                dist = np.stack(sd_results)
                dist_mask = prob

            prob = np.expand_dims(prob, -1)
            dist_mask = np.expand_dims(dist_mask, -1)

            # subsample wth given grid
            prob = prob[self.ss_grid]

            # append dist_mask to dist as additional channel
            dist_mask = dist_mask[self.ss_grid]
            dist = dist[self.ss_grid]
            dist = np.concatenate([dist, dist_mask], axis=-1)
            num_instances = [len(np.unique(el)) - 1 for el in Y]
            # pr.disable()
            # print(
            #     "time needed for live prep of training data: "
            #     f"{timeit.default_timer() - start_t:.3f}",
            # )
            # pr.print_stats(sort="time")
            # end_time = timeit.default_timer()
            # print(
                # f"time needed for post-proc calcs: {end_time - sample_patch_time:.3f}"
            # )
            # print(f"total time: {end_time - start_t:.3f}")
            # input()
            return [X], [prob, dist], num_instances
        return X, Y
