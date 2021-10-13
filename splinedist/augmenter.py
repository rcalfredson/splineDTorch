from csbdeep.utils import normalize
import cv2
import numpy as np
from PIL import Image
import random
import scipy.ndimage as ndi
from splinedist.config import Config
from splinedist.rotation import rotate_image
import timeit
import torch
import torchvision
from util import background_color, get_border_vals

def random_360rot(img: torch.Tensor, mask: torch.Tensor, background_color=0):
    rotation_angle = np.random.randint(0, 360)
    img = rotate_image(
        img,
        rotation_angle,
        use_linear=True,
        crop_based_on_orig_size=True,
        background_color=background_color,
    )[0]
    mask = rotate_image(
        mask, rotation_angle, use_linear=False, crop_based_on_orig_size=True
    )[0]
    return img, mask

def random_fliprot(img: torch.Tensor, mask: torch.Tensor):
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    # img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
    # mask = mask.transpose(perm)
    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask

def random_zoom(img: torch.Tensor, mask: torch.Tensor, config: Config):
        if config.zoom_min == None or config.zoom_max == None:
            return
        zoom_level = np.random.uniform(config.zoom_min, config.zoom_max)
        zoomed_img = ndi.zoom(img, zoom_level, order=3)
        zoomed_mask = ndi.zoom(mask, zoom_level, order=0)
        if zoom_level < 1:
            new_img = np.empty(img.shape, dtype=np.uint8)
            new_mask = np.zeros(mask.shape, dtype=np.uint8)
            new_img[: zoomed_img.shape[0], : zoomed_img.shape[1]] = zoomed_img
            new_img[:, zoomed_img.shape[1] :] = current_bg_color
            new_img[zoomed_img.shape[0] :, :] = current_bg_color
            new_mask[: zoomed_img.shape[0], : zoomed_img.shape[1]] = zoomed_mask
        elif zoom_level > 1:
            new_img = zoomed_img[: img.shape[0], : img.shape[1]]
            new_mask = zoomed_mask[: img.shape[0], : img.shape[1]]
        return (new_img, new_mask)

class Augmenter:
    def __init__(self, config, opts, axis_norm=(0, 1), normalize=True):
        self.color_jitter = torchvision.transforms.ColorJitter([0.6, 1], 0, 0.2, 0.05)
        self.random_shift = torchvision.transforms.RandomAffine(
            0, translate=(0.15, 0.15)
        )
        self.resize_edge_lengths = (640, 672, 704, 736, 768, 800)
        self.max_edge_length = 1333
        self.axis_norm = axis_norm
        self.config = config
        self.opts = opts
        self.normalize = normalize

    def add_color_jitter(self, x):
        x = x.astype(np.uint8)
        img = Image.fromarray(x)
        img = self.color_jitter(img)
        ret_arr = np.array(img)
        return ret_arr

    def add_random_shift(self, x, y):
        x = x.astype(np.uint8)
        img = Image.fromarray(x)
        mask = Image.fromarray(y)
        params = self.random_shift.get_params(
            (0, 0), (0.15, 0, 15), (1, 1), (0, 0), img.size
        )
        img = torchvision.transforms.functional.affine(
            img,
            *params,
            interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC,
            fill=current_bg_color,
        )
        shifted_mask = torchvision.transforms.functional.affine(
            mask,
            *params,
            interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST,
        )
        img = np.array(img)
        shifted_mask = np.array(shifted_mask)
        img = fill_in_blanks(img, current_bg_color)
        return img, y

    def add_clahe_contrast_adj(self, x):
        if np.random.random() < 0.3:
            return x
        try:
            x = x.astype(np.uint8)
            clipLimit = np.random.randint(1, 17)
            lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
            cl = clahe.apply(l.astype(np.uint8))
            limg = cv2.merge((cl, a, b))
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            return final
        except cv2.error as exc:
            return x

    def add_blur(self, x):
        ksize = random.randrange(1, 13, 2)
        x = cv2.GaussianBlur(x, (ksize, ksize), 1)
        return x

    def resize_shortest_edge(
        self, img, side_length=None, interpolation=cv2.INTER_CUBIC
    ):
        h, w = img.shape[:2]
        if side_length is None:
            side_length = np.random.choice(self.resize_edge_lengths)
        scale = side_length * 1 / min(h, w)
        if h < w:
            newh, neww = side_length, scale * w
        else:
            newh, neww = scale * h, side_length
        if max(newh, neww) > self.max_edge_length:
            scale = self.max_edge_length * 1 / max(newh, neww)

        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=interpolation)
        return img, side_length

    def augment(self, x, y):
        """Augmentation of a single input/label image pair.
        x is an input image
        y is the corresponding ground-truth label image
        """
        no_cutoff = None
        start_time = timeit.default_timer()
        x_orig = x
        y_orig = y
        num_attempts = 0
        while no_cutoff != True and num_attempts < 3:
            global current_bg_color
            current_bg_color = background_color(x_orig.astype(np.uint8))
            if no_cutoff != True and num_attempts < 2:
                # x, y = self.add_random_shift(x_orig, y_orig)
                pass
            # x, side_length = self.resize_shortest_edge(x_orig)
            # y, _ = self.resize_shortest_edge(
            # y_orig, side_length=side_length, interpolation=cv2.INTER_NEAREST
            # )
            # cv2.imshow('img_after', x.astype(np.uint8))
            # cv2.imshow('mask_after', y)
            # cv2.waitKey()
            # x, y = random_zoom(x, y, self.config)
            x = self.add_clahe_contrast_adj(x)
            x = self.add_color_jitter(x)
            x = self.add_blur(x)
            # x, y = random_fliprot(x, y)
            # x_alt, y_alt = random_360rot(x, y, current_bg_color)

            # x_alt = fill_in_blanks(x_alt, current_bg_color)
            # cv2.imshow("immediately before rotation", x)
            # cv2.imshow("immediately after rotation", x_alt)
            # print('current bg color:', current_bg_color)
            # cv2.waitKey(0)
            # x, y = random_360rot(x, y)
            # cv2.imshow('mask after rotation', 5*y)
            # cv2.waitKey(0)
            # x_prenorm = x
            if self.normalize:
                x = normalize(x, 1, 99.8, axis=self.axis_norm)
            # x_alt = normalize(x_alt, 1, 99.8, axis=axis_norm)
            # cv2.imshow("normal img post-normalization", x)
            # cv2.imshow("rotated img post-normalization", x_alt)
            # cv2.waitKey(0)
            # cv2.imshow('before norm:', x_prenorm)
            # cv2.imshow('after norm:', x)
            # cv2.waitKey(0)
            # input()
            sig = 0.02 * np.random.uniform(0, 1)
            x = x + sig * np.random.normal(0, 1, x.shape)
            # x_alt = x_alt + sig * np.random.normal(0, 1, x.shape)

            # x = np.clip(x, a_min=0, a_max=1)
            if self.config.skip_partials:
                border_vals = get_border_vals(y)
                if len(border_vals) > 1:
                    no_cutoff = False
                    x = x_orig
                    y = y_orig
                else:
                    no_cutoff = True
            else:
                no_cutoff = True
            num_attempts += 1
        if self.normalize and no_cutoff == False:
            x = normalize(x, 1, 99.8, axis=self.axis_norm)
        # final step before displaying the images:
        # scale down the mask.
        # cv2.imshow('mask before downscaling', y)
        # x = cv2.resize(
        #     x, (0, 0), fx=(1 / IMG_SCALING_FACTOR), fy=(1 / IMG_SCALING_FACTOR)
        # )
        # y = cv2.resize(
        #     y, (0, 0), fx=(1 / IMG_SCALING_FACTOR), fy=(1 / IMG_SCALING_FACTOR)
        # )
        # cv2.imshow('mask after rescaling', y)
        # y = cv2.medianBlur(y, 13)
        # cv2.imshow('mask after applying filter', y)
        # cv2.waitKey(0)
        if hasattr(self.opts, 'debug_vis') and self.opts.debug_vis:
            cv2.imshow("augmented image", x)
            # colormapped_y = cv2.applyColorMap(y, cv2.COLORMAP_JET)
            cv2.imshow("ground truth masks", 5 * y)
            # cv2.imshow("image with extra rotation", x_alt)
            # cv2.imshow("masks with extra rotation", y_alt)
            # print('augmented image:', x)
            # print('ground truth masks:', y)
            # with np.printoptions(threshold=sys.maxsize), open("debug1", "w") as f:
            #     print("ground truth:", y, file=f)
            cv2.waitKey(0)
        return x, y