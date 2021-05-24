from collections import namedtuple
import os
import timeit

from splinedist.models.backbone_types import BackboneTypes
from csbdeep.data import Normalizer, NoNormalizer, Resizer, NoResizer
from csbdeep.internals.predict import tile_iterator
from csbdeep.utils import axes_check_and_normalize, axes_dict, _raise, move_image_axes
import math
import numpy as np
import spline_generator as sg
from splinedist.config import Config
from splinedist.constants import DEVICE
from splinedist.geometry.geom2d import dist_to_coord, polygons_to_label
from splinedist.models.fcrn import FCRN_A
from splinedist.models.unet_block import UNet, UNetFromTF
from splinedist.nms import non_maximum_suppression
from splinedist.utils import _is_power_of_2, data_dir
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from typing import Tuple
import warnings

EPS = torch.finfo(torch.float32).eps


def square(x):
    """Compute element-wise square."""
    return torch.pow(x, 2)


def masked_loss(mask, n_params, penalty, reg_weight, norm_by_mask):
    def loss(y_true, y_pred):
        return penalty(y_true - y_pred)

    return generic_masked_loss(
        mask, n_params, loss, reg_weight=reg_weight, norm_by_mask=norm_by_mask
    )


def masked_loss_mae(mask, n_params, reg_weight=0, norm_by_mask=True):
    return masked_loss(
        mask, n_params, torch.abs, reg_weight=reg_weight, norm_by_mask=norm_by_mask
    )


def masked_loss_mse(mask, n_params, reg_weight=0, norm_by_mask=True):
    return masked_loss(
        mask, n_params, square, reg_weight=reg_weight, norm_by_mask=norm_by_mask
    )


def masked_metric_mae(mask, n_params):
    def relevant_mae(y_true, y_pred):
        return masked_loss(mask, n_params, torch.abs, reg_weight=0, norm_by_mask=True)(
            y_true, y_pred
        )

    return relevant_mae


def masked_metric_mse(mask, n_params):
    def relevant_mse(y_true, y_pred):
        return masked_loss(mask, n_params, square, reg_weight=0, norm_by_mask=True)(
            y_true, y_pred
        )

    return relevant_mse


def generic_masked_loss(
    mask,
    n_params,
    loss,
    weights=1,
    norm_by_mask=True,
    reg_weight=0,
    reg_penalty=torch.abs,
):
    def _loss(y_true, y_pred):
        y_pred = torch.reshape(
            y_pred,
            (y_pred.shape[0], y_pred.shape[1], y_pred.shape[2], -1, 2),
        )
        y_pred_r = y_pred[:, :, :, :, 0]
        y_pred_theta = y_pred[:, :, :, :, 1]

        x = y_pred_r * torch.cos(y_pred_theta)
        y = y_pred_r * torch.sin(y_pred_theta)
        y_pred = torch.stack((x, y), dim=-1)

        M = n_params // 2
        grid = np.load(os.path.join(data_dir(), "grid_" + str(M) + ".npy"))
        grid = torch.from_numpy(grid).float().to(DEVICE)
        grid = torch.repeat_interleave(grid, y_pred.shape[0], dim=0)
        c_pred = grid + y_pred

        phi = np.load(os.path.join(data_dir(), "phi_" + str(M) + ".npy"))
        phi = torch.from_numpy(phi).float().to(DEVICE)
        SplineContour = sg.SplineCurveVectorized(M, sg.B3(), True, c_pred)
        y_pred = SplineContour.sampleSequential(phi)
        y_pred = torch.reshape(
            y_pred, (y_pred.shape[0], y_pred.shape[1], y_pred.shape[2], -1)
        )
        actual_loss = torch.mean(mask * weights * loss(y_true, y_pred), dim=-1)
        norm_mask = (torch.mean(mask) + EPS) if norm_by_mask else 1
        if reg_weight > 0:
            reg_loss = torch.mean((1 - mask) * reg_penalty(y_pred), dim=-1)
            return torch.mean(actual_loss / norm_mask + reg_weight * reg_loss)
        else:
            return torch.mean(actual_loss / norm_mask)

    return _loss


class SamePadder(nn.Module):
    def __init__(self, filter_shape):
        super(SamePadder, self).__init__()
        self.filter_shape = filter_shape

    def forward(self, input):
        strides = (None, 1, 1)
        in_height, in_width = input.shape[1:3]
        filter_height, filter_width = self.filter_shape

        if in_height % strides[1] == 0:
            pad_along_height = max(filter_height - strides[1], 0)
        else:
            pad_along_height = max(filter_height - (in_height % strides[1]), 0)
        if in_width % strides[2] == 0:
            pad_along_width = max(filter_width - strides[2], 0)
        else:
            pad_along_width = max(filter_width - (in_width % strides[2]), 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom))


class ConvCat(nn.Module):
    """Convolution with upsampling + concatenate block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        size: Tuple[int, int],
        N,
        stride: Tuple[int, int] = (1, 1),
    ):
        """
        Create a sequential container with convolutional block (see conv_block)
        with N convolutional layers and upsampling by factor 2.
        """
        super(ConvCat, self).__init__()
        self.conv = nn.Sequential(
            conv_block(in_channels, out_channels, size, stride, N),
            nn.Upsample(scale_factor=2),
        )

    def forward(self, to_conv: torch.Tensor, to_cat: torch.Tensor):
        """Forward pass.

        Args:
            to_conv: input passed to convolutional block and upsampling
            to_cat: input concatenated with the output of a conv block
        """
        return torch.cat([self.conv(to_conv), to_cat], dim=1)


def conv_block(in_channels, out_channels, kernel_size, N, stride=1, activation="relu"):
    activation = {"relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "linear": None}[
        activation.lower()
    ]

    def block(in_channels):
        steps = [
            SamePadder(kernel_size),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
            ),
        ]
        if activation is not None:
            steps += [activation]
        return nn.Sequential(*steps)

    obj = nn.Sequential(
        *[block(in_channels if not i else out_channels) for i in range(N)]
    )
    return obj


class SplineDist2D(nn.Module):
    def __init__(self, config: Config, train=True):
        super(SplineDist2D, self).__init__()
        self.config = config
        self.add_subsampling_layers()
        if self.config.backbone == BackboneTypes.unet:
            self.backbone_block = UNet(32, 1, bilinear=False)
            # self.backbone_block = UNetFromTF(32, 1)
        elif self.config.backbone == BackboneTypes.fcrn_a:
            self.backbone_block = FCRN_A(2, 32)
        self.add_post_backbone_block()
        self.add_output_layers()
        self.thresholds = namedtuple("Thresholds", ("prob", "nms"))(0.5, 0.4)
        if train:
            self.prepare_for_training()

    def prepare_for_training(self):
        masked_dist_loss = {"mse": masked_loss_mse, "mae": masked_loss_mae}[
            self.config.train_dist_loss
        ]

        def split_dist_true_mask(dist_true_mask):
            # return tf.split(dist_true_mask, num_or_size_splits=[self.config.n_params,-1], axis=-1)
            return torch.split(
                dist_true_mask,
                split_size_or_sections=[2 * self.config.contoursize_max, 1],
                dim=-1,
            )

        def dist_loss(dist_pred, dist_true_mask):
            dist_true, dist_mask = split_dist_true_mask(dist_true_mask)
            return masked_dist_loss(
                dist_mask,
                n_params=self.config.n_params,
                reg_weight=self.config.train_background_reg,
            )(dist_true, dist_pred)

        self.dist_loss = dist_loss
        self.prob_loss = torch.nn.BCELoss()

        def loss(true, pred):
            return sum(
                [
                    loss_fn(pred[i], true[i]) * self.config.train_loss_weights[i]
                    for i, loss_fn in enumerate((self.prob_loss, self.dist_loss))
                ]
            )

        self.loss = loss

    def add_post_backbone_block(self):
        self.post_backbone_block = conv_block(
            self.config.n_filter_base,
            self.config.net_conv_after_backbone,
            self.config.kernel_size,
            1,
        )

    def add_output_layers(self):
        self.output_prob_layer = conv_block(
            4 * self.config.n_filter_base, 1, (1, 1), 1, 1, "sigmoid"
        )
        self.output_dist_layer = conv_block(
            4 * self.config.n_filter_base,
            self.config.n_params,
            (1, 1),
            1,
            1,
            "linear",
        )

    def _make_permute_axes(
        self, img_axes_in, net_axes_in, net_axes_out=None, img_axes_out=None
    ):
        # img_axes_in -> net_axes_in ---NN--> net_axes_out -> img_axes_out
        if net_axes_out is None:
            net_axes_out = net_axes_in
        if img_axes_out is None:
            img_axes_out = img_axes_in
        assert "C" in net_axes_in and "C" in net_axes_out
        assert not "C" in img_axes_in or "C" in img_axes_out

        def _permute_axes(data, undo=False):
            if data is None:
                return None
            if undo:
                if "C" in img_axes_in:
                    return move_image_axes(data, net_axes_out, img_axes_out, True)
                else:
                    # input is single-channel and has no channel axis
                    data = move_image_axes(data, net_axes_out, img_axes_out + "C", True)
                    if data.shape[-1] == 1:
                        # output is single-channel -> remove channel axis
                        data = data[..., 0]
                    return data
            else:
                return move_image_axes(data, img_axes_in, net_axes_in, True)

        return _permute_axes

    def _normalize_axes(self, img, axes):
        if axes is None:
            axes = self.config.axes
            assert "C" in axes
            if img.ndim == len(axes) - 1 and self.config.n_channel_in == 1:
                # img has no dedicated channel axis, but 'C' always part of config axes
                axes = axes.replace("C", "")
        return axes_check_and_normalize(axes, img.ndim)

    def _compute_receptive_field(self, img_size=None):
        # TODO: good enough?
        from scipy.ndimage import zoom

        if img_size is None:
            img_size = tuple(
                g * (128 if self.config.n_dim == 2 else 64) for g in self.config.grid
            )
        if np.isscalar(img_size):
            img_size = (img_size,) * self.config.n_dim
        img_size = tuple(img_size)
        assert all(_is_power_of_2(s) for s in img_size)
        mid = tuple(s // 2 for s in img_size)
        x = np.zeros((1,) + img_size + (self.config.n_channel_in,), dtype=np.float32)
        z = np.zeros_like(x)
        x[(0,) + mid + (slice(None),)] = 1
        x = torch.from_numpy(x).permute(0, 3, 1, 2).cuda()
        z = torch.from_numpy(z).permute(0, 3, 1, 2).cuda()
        result = self(x)
        y = result[0][0, 0, ...]
        y0 = self(z)[0][0, 0, ...]
        grid = tuple((np.array(x.shape[-2:-1]) / np.array(y.shape)).astype(int))
        assert grid == self.config.grid
        y, y0 = y.cpu().detach().numpy(), y0.cpu().detach().numpy()
        y = zoom(y, grid, order=0)
        y0 = zoom(y0, grid, order=0)
        ind = np.where(np.abs(y - y0) > 0)
        return [(m - np.min(i), np.max(i) - m) for (m, i) in zip(mid, ind)]

    def add_subsampling_layers(self):
        self.subsample_convs = nn.ModuleList()
        self.pool_kernels = []
        pooled = np.array([1, 1])
        while tuple(pooled) != tuple(self.config.grid):
            pool = 1 + (np.asarray(self.config.grid) > pooled)
            pooled *= pool
            self.pool_kernels.append(tuple(pool))
            self.subsample_convs.append(
                conv_block(
                    in_channels=self.config.n_channel_in,
                    out_channels=self.config.n_filter_base,
                    kernel_size=self.config.kernel_size,
                    N=self.config.n_conv_per_depth,
                )
            )

    def forward(self, input: torch.Tensor):
        data = input
        for i, conv in enumerate(self.subsample_convs):
            pool = nn.MaxPool2d(self.pool_kernels[i])
            data = conv(data)
            data = pool(data)
        data = self.backbone_block(data)
        data = self.post_backbone_block(data)
        output_prob = self.output_prob_layer(data)
        output_dist = self.output_dist_layer(data)
        return [output_prob, output_dist]

    def _axes_tile_overlap(self, query_axes):
        query_axes = axes_check_and_normalize(query_axes)
        try:
            self._tile_overlap
        except AttributeError:
            self._tile_overlap = self._compute_receptive_field()
        overlap = dict(
            zip(
                self.config.axes.replace("C", ""),
                tuple(max(rf) for rf in self._tile_overlap),
            )
        )
        return tuple(overlap.get(a, 0) for a in query_axes)

    def _axes_div_by(self, query_axes):
        self.config.backbone in BackboneTypes or _raise(NotImplementedError())
        query_axes = axes_check_and_normalize(query_axes)
        assert len(self.config.pool) == len(self.config.grid)
        div_by = dict(
            zip(
                self.config.axes.replace("C", ""),
                tuple(
                    p ** self.config.n_depth * g
                    for p, g in zip(self.config.pool, self.config.grid)
                ),
            )
        )
        return tuple(div_by.get(a, 1) for a in query_axes)

    def _check_normalizer_resizer(self, normalizer, resizer):
        if normalizer is None:
            normalizer = NoNormalizer()
        if resizer is None:
            resizer = NoResizer()
        isinstance(resizer, Resizer) or _raise(ValueError())
        isinstance(normalizer, Normalizer) or _raise(ValueError())
        if normalizer.do_after:
            if self.config.n_channel_in != self.config.n_channel_out:
                warnings.warn(
                    "skipping normalization step after prediction because "
                    + "number of input and output channels differ."
                )

        return normalizer, resizer

    def predict(
        self,
        img,
        axes=None,
        normalizer=None,
        n_tiles=None,
        show_tile_progress=True,
        **predict_kwargs
    ):
        # total_cuda_time = 0
        if n_tiles is None:
            n_tiles = [1] * img.ndim
        try:
            n_tiles = tuple(n_tiles)
            img.ndim == len(n_tiles) or _raise(TypeError())
        except TypeError:
            raise ValueError("n_tiles must be an iterable of length %d" % img.ndim)
        all(np.isscalar(t) and 1 <= t and int(t) == t for t in n_tiles) or _raise(
            ValueError("all values of n_tiles must be integer values >= 1")
        )
        n_tiles = tuple(map(int, n_tiles))

        axes = self._normalize_axes(img, axes)
        axes_net = self.config.axes

        _permute_axes = self._make_permute_axes(axes, axes_net)
        x = _permute_axes(img)  # x has axes_net semantics

        channel = axes_dict(axes_net)["C"]
        self.config.n_channel_in == x.shape[channel] or _raise(ValueError())
        axes_net_div_by = self._axes_div_by(axes_net)

        grid = tuple(self.config.grid)
        len(grid) == len(axes_net) - 1 or _raise(ValueError())
        grid_dict = dict(zip(axes_net.replace("C", ""), grid))

        normalizer = self._check_normalizer_resizer(normalizer, None)[0]
        resizer = SplineDistPadAndCropResizer(grid=grid_dict)

        x = normalizer.before(x, axes_net)
        x = resizer.before(x, axes_net, axes_net_div_by)

        def predict_direct(tile: np.ndarray):
            # nonlocal total_cuda_time
            # start_t = timeit.default_timer()
            tile = tile[np.newaxis]
            prob, dist = self(torch.from_numpy(tile).permute(0, 3, 1, 2).cuda())
            # total_cuda_time += timeit.default_timer() - start_t
            return prob[0].permute(1, 2, 0), dist[0].permute(1, 2, 0)

        if np.prod(n_tiles) > 1:
            tiling_axes = axes_net.replace("C", "")  # axes eligible for tiling
            x_tiling_axis = tuple(
                axes_dict(axes_net)[a] for a in tiling_axes
            )  # numerical axis ids for x
            axes_net_tile_overlaps = self._axes_tile_overlap(axes_net)
            # hack: permute tiling axis in the same way as img -> x was permuted
            n_tiles = _permute_axes(np.empty(n_tiles, np.bool)).shape
            (
                all(n_tiles[i] == 1 for i in range(x.ndim) if i not in x_tiling_axis)
                or _raise(
                    ValueError(
                        "entry of n_tiles > 1 only allowed for axes '%s'" % tiling_axes
                    )
                )
            )

            sh = [s // grid_dict.get(a, 1) for a, s in zip(axes_net, x.shape)]
            sh[channel] = 1
            prob = np.empty(sh, np.float32)
            sh[channel] = self.config.n_params
            dist = np.empty(sh, np.float32)

            n_block_overlaps = [
                int(np.ceil(overlap / blocksize))
                for overlap, blocksize in zip(axes_net_tile_overlaps, axes_net_div_by)
            ]

            for tile, s_src, s_dst in tqdm(
                tile_iterator(
                    x,
                    n_tiles,
                    block_sizes=axes_net_div_by,
                    n_block_overlaps=n_block_overlaps,
                ),
                disable=(not show_tile_progress),
                total=np.prod(n_tiles),
            ):
                prob_tile, dist_tile = predict_direct(tile)
                # account for grid
                s_src = [
                    slice(s.start // grid_dict.get(a, 1), s.stop // grid_dict.get(a, 1))
                    for s, a in zip(s_src, axes_net)
                ]
                s_dst = [
                    slice(s.start // grid_dict.get(a, 1), s.stop // grid_dict.get(a, 1))
                    for s, a in zip(s_dst, axes_net)
                ]
                # prob and dist have different channel dimensionality than image x
                s_src[channel] = slice(None)
                s_dst[channel] = slice(None)
                s_src, s_dst = tuple(s_src), tuple(s_dst)
                # print(s_src,s_dst)
                prob[s_dst] = prob_tile[s_src]
                dist[s_dst] = dist_tile[s_src]

        else:
            prob, dist = predict_direct(x)

        # start_t = timeit.default_timer()
        prob = resizer.after(prob, axes_net)
        dist = resizer.after(dist, axes_net)
        # total_cuda_time += timeit.default_timer() - start_t
        prob = prob.cpu().detach().numpy()
        dist = dist.cpu().detach().numpy()

        prob = np.take(prob, 0, axis=channel)
        dist = np.moveaxis(dist, channel, -1)
        # print('total cuda time:', total_cuda_time)
        return prob, dist

    def predict_instances(
        self,
        img,
        axes=None,
        normalizer=None,
        prob_thresh=None,
        nms_thresh=None,
        n_tiles=None,
        show_tile_progress=True,
        verbose=False,
        predict_kwargs=None,
        nms_kwargs=None,
        overlap_label=None,
    ):
        """Predict instance segmentation from input image.

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Input image
        axes : str or None
            Axes of the input ``img``.
            ``None`` denotes that axes of img are the same as denoted in the config.
        normalizer : :class:`csbdeep.data.Normalizer` or None
            (Optional) normalization of input image before prediction.
            Note that the default (``None``) assumes ``img`` to be already normalized.
        prob_thresh : float or None
            Consider only object candidates from pixels with predicted object probability
            above this threshold (also see `optimize_thresholds`).
        nms_thresh : float or None
            Perform non-maximum suppression that considers two objects to be the same
            when their area/surface overlap exceeds this threshold (also see `optimize_thresholds`).
        n_tiles : iterable or None
            Out of memory (OOM) errors can occur if the input image is too large.
            To avoid this problem, the input image is broken up into (overlapping) tiles
            that are processed independently and re-assembled.
            This parameter denotes a tuple of the number of tiles for every image axis (see ``axes``).
            ``None`` denotes that no tiling should be used.
        show_tile_progress: bool
            Whether to show progress during tiled prediction.
        predict_kwargs: dict
            Keyword arguments for ``predict`` function of Keras model.
        nms_kwargs: dict
            Keyword arguments for non-maximum suppression.
        overlap_label: scalar or None
            if not None, label the regions where polygons overlap with that value

        Returns
        -------
        (:class:`numpy.ndarray`, dict)
            Returns a tuple of the label instances image and also
            a dictionary with the details (coordinates, etc.) of all remaining polygons/polyhedra.

        """
        # start_t = timeit.default_timer()
        if predict_kwargs is None:
            predict_kwargs = {}
        if nms_kwargs is None:
            nms_kwargs = {}

        nms_kwargs.setdefault("verbose", verbose)

        _axes = self._normalize_axes(img, axes)
        _axes_net = self.config.axes
        _permute_axes = self._make_permute_axes(_axes, _axes_net)
        _shape_inst = tuple(
            s for s, a in zip(_permute_axes(img).shape, _axes_net) if a != "C"
        )

        prob, dist = self.predict(
            img,
            axes=axes,
            normalizer=normalizer,
            n_tiles=n_tiles,
            show_tile_progress=show_tile_progress,
            **predict_kwargs
        )
        finals = self._instances_from_prediction(
            _shape_inst,
            prob,
            dist,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            overlap_label=overlap_label,
            **nms_kwargs
        )
        # print('total predict time:', timeit.default_timer() - start_t)
        return finals

    def _instances_from_prediction(
        self,
        img_shape,
        prob,
        dist,
        prob_thresh=None,
        nms_thresh=None,
        overlap_label=None,
        **nms_kwargs
    ):
        if prob_thresh is None:
            prob_thresh = self.thresholds.prob
        if nms_thresh is None:
            nms_thresh = self.thresholds.nms
        if overlap_label is not None:
            raise NotImplementedError("overlap_label not supported for 2D yet!")

        coord = dist_to_coord(dist, grid=self.config.grid)
        inds = non_maximum_suppression(
            coord,
            prob,
            grid=self.config.grid,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            **nms_kwargs
        )
        labels = polygons_to_label(coord, prob, inds, shape=img_shape)
        # sort 'inds' such that ids in 'labels' map to entries in polygon dictionary entries
        inds = inds[np.argsort(prob[inds[:, 0], inds[:, 1]])]
        # adjust for grid
        points = inds * np.array(self.config.grid)
        return (
            labels,
            dict(
                coord=coord[inds[:, 0], inds[:, 1]],
                points=points,
                prob=prob[inds[:, 0], inds[:, 1]],
            ),
        )


class SplineDistPadAndCropResizer(Resizer):

    # TODO: check correctness
    def __init__(self, grid, mode="reflect", **kwargs):
        assert isinstance(grid, dict)
        self.mode = mode
        self.grid = grid
        self.kwargs = kwargs

    def before(self, x, axes, axes_div_by):
        assert all(
            a % g == 0 for g, a in zip((self.grid.get(a, 1) for a in axes), axes_div_by)
        )
        axes = axes_check_and_normalize(axes, x.ndim)

        def _split(v):
            return 0, v  # only pad at the end

        self.pad = {
            a: _split((div_n - s % div_n) % div_n)
            for a, div_n, s in zip(axes, axes_div_by, x.shape)
        }
        x_pad = np.pad(
            x, tuple(self.pad[a] for a in axes), mode=self.mode, **self.kwargs
        )
        self.padded_shape = dict(zip(axes, x_pad.shape))
        if "C" in self.padded_shape:
            del self.padded_shape["C"]
        return x_pad

    def after(self, x, axes):
        # axes can include 'C', which may not have been present in before()
        axes = axes_check_and_normalize(axes, x.ndim)
        assert all(
            s_pad == s * g
            for s, s_pad, g in zip(
                x.shape,
                (self.padded_shape.get(a, _s) for a, _s in zip(axes, x.shape)),
                (self.grid.get(a, 1) for a in axes),
            )
        )

        crop = tuple(
            slice(0, -(math.floor(p[1] / g)) if p[1] >= g else None)
            for p, g in zip(
                (self.pad.get(a, (0, 0)) for a in axes),
                (self.grid.get(a, 1) for a in axes),
            )
        )
        return x[crop]
