"""Main script used to train networks.
"""
import argparse
from detectron2.data import transforms as T
from splinedist.constants import DEVICE
from matplotlib import pyplot as plt
from numpy.core.defchararray import mod
import torch
import torchvision.transforms
import torchvision.transforms.functional
from csbdeep.utils import normalize
import cv2
import datetime
import functools
from glob import glob
import json
import math
import numpy as np
import os
from pathlib import Path
from PIL import Image, ImageDraw
import platform
from pycocotools.coco import COCO
import random
import shutil
import signal
from splinedist.config import Config
from splinedist.rotation import rotate_image
from splinedist.looper import Looper
from splinedist.models.model2d import SplineDist2D
from splinedist.utils import (
    calculate_extents,
    data_dir,
    fill_label_holes,
    get_contoursize_max,
    grid_generator,
    phi_generator,
)
import sys
from tifffile import imread
import timeit
from util import background_color, fill_in_blanks, get_border_vals

import scipy.ndimage as ndi
from tqdm import tqdm

# python trainByBatchLinux.py 10 "--config configs/unet_backbone_rand_zoom.json --plot --val_interval 4"

sys_type = platform.system()
win_drive_letter = "P"
if sys_type == "Windows" and "Dell2" in platform.node():
    win_drive_letter = "R"
DATA_BASE_DIR = {
    "Windows": f"{win_drive_letter}:/Robert/splineDist/data",
    "Linux": "/media/Synology3/Robert/splineDist/data",
}[sys_type]
train_ts = datetime.datetime.now()

# IMG_SCALING_FACTOR = 2.5

lr_history = {}


def options():
    parser = argparse.ArgumentParser(
        description="Train object detection model using SplineDist"
    )
    parser.add_argument(
        "data_path",
        help='path to the folder containing the "images" and "masks" folders with the '
        "data to use during training. Must be specified relative to"
        "the --data_base_dir.",
    )
    parser.add_argument(
        "--data_base_dir",
        help="path to the folder where individual datasets are"
        " stored (as specified by data_path). Defaults to Robert/splineDist/data"
        " on Synology3.",
        default=DATA_BASE_DIR,
    )
    parser.add_argument(
        "--coco_file_path",
        help="path to COCO file containing annotation data that corresponds to the"
        " images used during training, which are used to generate masks dynamically"
        " in place of the masks stored in the folder associated with 'data_path.'",
    )
    parser.add_argument(
        "--config",
        help="Config file to use (default: configs/defaults.json).",
        default="configs/defaults.json",
    )
    parser.add_argument(
        "-m", "--model_path", default="", help="Path of an existing model to load"
    )
    parser.add_argument(
        "--init_dists",
        help="path to a JSON file used to initialize the weights of"
        " the neural net. Its keys are names of net layers, and its values are"
        ' dicts of the form {"mean": float, "var": float}, describing a Gaussian.'
        " The weights of any layer not names in ther",
    )
    parser.add_argument("--plot", help="Add plots", action="store_true")
    parser.add_argument(
        "--left_col_plots",
        choices=["mae", "scatter"],
        default="mae",
        help="Plot type to be displayed in the left column: mae (mean absolute error) or "
        "scatter (scatter plot of individual predictions from the most recent epoch)",
    )
    parser.add_argument(
        "--export_at_end",
        action="store_true",
        help="Save the model and the plot from the final epoch of training only, with"
        " the timestamp of the training's end in their filenames.",
    )
    parser.add_argument(
        "--val_interval",
        default=20,
        type=int,
        help="Frequency, in epochs, at which to measure validation performance.",
    )
    parser.add_argument(
        "--debug",
        help="1) print additional information used for debugging trainings and 2) save"
        " artifacts in a separate folder labeled 'debug'",
        action="store_true",
    )
    parser.add_argument(
        "--debug_vis",
        help="show image and segmentation masks in a live viewer for debugging",
        action="store_true",
    )

    return parser.parse_args()


opts = options()
# create a folder for artifacts for this machine if needed
results_for_host = f"./results_{platform.node()}"
if opts.debug:
    results_for_host = os.path.join(results_for_host, "debug")
Path(results_for_host).mkdir(exist_ok=True, parents=True)
if not os.path.isdir(data_dir(must_exist=False)):
    Path(data_dir(must_exist=False)).mkdir(parents=True, exist_ok=True)
X = sorted(glob(os.path.join(DATA_BASE_DIR, opts.data_path, "images/*.tif")))
img_filepaths = X
masks = {k: [] for k in X}
# X = [cv2.resize(img, (0, 0), fx=IMG_SCALING_FACTOR, fy=IMG_SCALING_FACTOR) for img in list(map(imread, X))]
X = list(map(imread, X))


def create_masks(filepath, coco_data, img_id, orig_img, num_masks=6):
    img_data = coco_data.imgs[img_id]
    for _ in range(num_masks):
        maskImg = Image.new(
            "L",
            (
                # IMG_SCALING_FACTOR * img_data["width"],
                # IMG_SCALING_FACTOR * img_data["height"],
                img_data["width"],
                img_data["height"],
            ),
            0,
        )
        annotations = coco_data.getAnnIds(imgIds=[img_id])
        random.shuffle(annotations)
        for i, ann in enumerate(annotations):
            ann = coco_data.anns[ann]
            ImageDraw.Draw(maskImg).polygon(
                # [int(IMG_SCALING_FACTOR * el) for el in ann["segmentation"][0]],
                [int(el) for el in ann["segmentation"][0]],
                outline=i + 1,
                fill=i + 1,
            )
            if len(ann["segmentation"]) > 1:
                for seg in ann["segmentation"][1:]:
                    ImageDraw.Draw(maskImg).polygon(
                        # [int(IMG_SCALING_FACTOR * el) for el in seg], outline=0, fill=0
                        [int(el) for el in seg],
                        outline=0,
                        fill=0,
                    )
        maskImg = np.array(maskImg)
        masks[filepath].append(maskImg)


if opts.coco_file_path:
    coco_data = COCO(opts.coco_file_path)
    for i, filepath in enumerate(img_filepaths):
        img_basename = os.path.basename(filepath)

        for img_id in coco_data.imgs:
            if (
                os.path.splitext(coco_data.imgs[img_id]["file_name"])[0]
                == os.path.splitext(img_basename)[0]
            ):
                create_masks(filepath, coco_data, img_id, X[i])
else:
    mask_filepaths = sorted(
        glob(os.path.join(DATA_BASE_DIR, opts.data_path, "masks/*.tif"))
    )
    for filepath in img_filepaths:
        img_basename = os.path.basename(filepath)
        mask_filepath = os.path.join(
            DATA_BASE_DIR, opts.data_path, "masks", img_basename
        )
        assert os.path.exists(mask_filepath)
        # masks[filepath].append(cv2.resize(imread(mask_filepath), (0, 0),
        #     fx=IMG_SCALING_FACTOR, fy=IMG_SCALING_FACTOR,
        #     interpolation=cv2.INTER_NEAREST))
        masks[filepath].append(imread(mask_filepath))
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
axis_norm = (0, 1)  # normalize channels independently

if n_channel > 1:
    print(
        "Normalizing image channels %s."
        % ("jointly" if axis_norm is None or 2 in axis_norm else "independently")
    )
    sys.stdout.flush()

# X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X)]
for k in masks:
    for i, arr in enumerate(masks[k]):
        masks[k][i] = fill_label_holes(arr)

assert len(X) > 1, "not enough training data"
rng = np.random.RandomState(seed=None)  # 42)

ind = rng.permutation(len(X))
n_val = max(1, int(round(0.21 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]

X_val, Y_val = [X[i] for i in ind_val], [masks[img_filepaths[i]] for i in ind_val]
X_trn, Y_trn = [X[i] for i in ind_train], [masks[img_filepaths[i]] for i in ind_train]
filenames_trn = [img_filepaths[i] for i in ind_train]
filenames_val = [img_filepaths[i] for i in ind_val]
print("number of images: %3d" % len(X))
print("- training:       %3d" % len(X_trn))
print("- validation:     %3d" % len(X_val))

i = min(9, len(X) - 1)
img, lbl = X[i], masks[img_filepaths[0]]
assert img.ndim in (2, 3)
img = img if (img.ndim == 2 or img.shape[-1] == 3) else img[..., 0]

M = 8
n_params = 2 * M
grid = (2, 2)

contoursize_max = get_contoursize_max(
    [mask for sublist in masks.values() for mask in sublist]
)
config_kwargs = {"n_channel_in": n_channel, "contoursize_max": contoursize_max}
config = Config(opts.config, **config_kwargs)
if opts.plot:
    plt.ion()
    fig, plots = plt.subplots(nrows=2, ncols=2)
else:
    plots = [None] * 2

phi_generator(M, int(config.contoursize_max))
for patch_size in (
    (640, 960),
    (672, 1008),
    (704, 1056),
    (736, 1104),
    (768, 1152),
    (800, 1200),
):
    grid_generator(M, patch_size, config.grid)
current_bg_color = None


def write_existing_model_data():
    filename = (
        f"splinedist_{config.backbone.name}_metadata"
        f"_{platform.node()}_{train_ts}.json".replace(":", "-")
    )
    metadata = {"starter_model": opts.model_path}
    with open(os.path.join(results_for_host, filename), "w") as f:
        json.dump(metadata, f)


def write_lr_history(config: Config):
    if not hasattr(config, "lr_history_filename"):
        setattr(
            config,
            "lr_history_filename",
            f"splinedist_{config.backbone.name}_lr_history"
            f"_{platform.node()}_{train_ts}.json".replace(":", "-"),
        )
    with open(os.path.join(results_for_host, config.lr_history_filename), "w") as my_f:
        json.dump(lr_history, my_f, ensure_ascii=False, indent=4)


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


def random_intensity_change(img):
    img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
    return img


def random_zoom(img: torch.Tensor, mask: torch.Tensor):
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
    def __init__(self):
        self.color_jitter = torchvision.transforms.ColorJitter([0.6, 1], 0, 0.2, 0.05)
        self.random_shift = torchvision.transforms.RandomAffine(
            0, translate=(0.15, 0.15)
        )
        self.resize_edge_lengths = (640, 672, 704, 736, 768, 800)
        self.max_edge_length = 1333

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
            clipLimit = np.random.randint(1, 17)
            lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
            cl = clahe.apply(l.astype(np.uint8))
            limg = cv2.merge((cl, a, b))
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            return final
        except cv2.error:
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
            # x, y = random_zoom(x, y)
            # x = self.add_clahe_contrast_adj(x)
            # x = self.add_color_jitter(x)
            # x = self.add_blur(x)
            x, y = random_fliprot(x, y)
            # x_alt, y_alt = random_360rot(x, y, current_bg_color)

            # x_alt = fill_in_blanks(x_alt, current_bg_color)
            # cv2.imshow("immediately before rotation", x)
            # cv2.imshow("immediately after rotation", x_alt)
            # print('current bg color:', current_bg_color)
            # cv2.waitKey(0)
            x, y = random_360rot(x, y)
            # x_prenorm = x
            x = normalize(x, 1, 99.8, axis=axis_norm)
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
            if config.skip_partials:
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
        if no_cutoff == False:
            x = normalize(x, 1, 99.8, axis=axis_norm)
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
        if opts.debug_vis:
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


model = SplineDist2D(config)
model.cuda()
learning_rate = config.train_learning_rate
if opts.model_path != "":
    model.load_state_dict(torch.load(opts.model_path))
    write_existing_model_data()
    parent_dir = Path(opts.model_path).parents[1]
    search_string = "_".join(opts.model_path.split("_")[-2:]).split(".pth")[0]
    lr_files = glob(os.path.join(parent_dir, "*lr_history*%s*" % search_string)) + glob(
        os.path.join(parent_dir, "*%s*lr_history*" % search_string)
    )
    if len(lr_files) > 0:
        with open(lr_files[0]) as f:
            lrs = json.load(f)
        learning_rate = list(lrs.values())[-1]
if opts.init_dists is not None:
    with open(opts.init_dists) as f:
        init_dists = json.load(f)
    for n, module in model.named_modules():
        layer_name = ".".join(n.split(".")[:2])
        if layer_name not in init_dists:
            continue
        try:
            with torch.no_grad():
                module.weight = torch.nn.Parameter(
                    torch.normal(
                        init_dists[layer_name]["mean"]
                        * torch.ones(module.weight.shape),
                        math.sqrt(init_dists[layer_name]["var"])
                        * torch.ones(module.weight.shape),
                    )
                )
        except AttributeError as exc:
            pass
model.cuda()
median_size = calculate_extents(
    [mask for sublist in masks.values() for mask in sublist], np.median
)
fov = np.array(model._axes_tile_overlap("YX"))
print(f"median object size:      {median_size}")
print(f"network field of view:   {fov}")
if any(median_size > fov):
    print(
        "WARNING: median object size larger than field of view of the neural network."
    )

optimizer = torch.optim.Adam(model.parameters(), learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=config.lr_reduct_factor, verbose=True, patience=config.lr_patience
)
augmenter = Augmenter()
model.prepare_for_training()

train_looper = Looper(
    model,
    config,
    DEVICE,
    model.loss,
    optimizer,
    augmenter.augment,
    X_trn,
    Y_trn,
    filenames_trn,
    validation=False,
    plots=plots[0],
    left_col_plots=opts.left_col_plots,
)
valid_looper = Looper(
    model,
    config,
    DEVICE,
    model.loss,
    optimizer,
    augmenter.augment,
    X_val,
    Y_val,
    filenames_val,
    validation=True,
    plots=plots[1],
    left_col_plots=opts.left_col_plots,
)

torch.save(
    model.state_dict(),
    os.path.join(
        results_for_host,
        f"splinedist_{config.backbone.name}_{train_ts}_pre_training"
        f"_{platform.node()}.pth".replace(":", "-"),
    ),
)

current_best = np.infty
for epoch in range(config.train_epochs):
    start_time = timeit.default_timer()
    current_lr = optimizer.param_groups[0]["lr"]
    if len(lr_history) == 0 or lr_history["last"] != current_lr:
        lr_history["last"] = current_lr
        lr_history[f"epoch_{epoch+1}"] = current_lr
        write_lr_history(config)
    start_time = timeit.default_timer()
    print(f"Epoch {epoch + 1}\n")

    train_looper.run(epoch)
    if epoch % opts.val_interval == 0:
        with torch.no_grad():
            val_loss = valid_looper.run(epoch)
        lr_scheduler.step(val_loss)
        new_best = val_loss < current_best
        reachedSaveInterval = epoch % 1 == 0
        if new_best:
            current_best = val_loss
            for f in glob(
                os.path.join(
                    results_for_host,
                    f"splinedist_{config.backbone.name}_{train_ts}_best_"
                    f"*_{platform.node()}.pth".replace(":", "-"),
                )
            ):
                os.unlink(f)
            torch.save(
                model.state_dict(),
                os.path.join(
                    results_for_host,
                    f"splinedist_{config.backbone.name}_{train_ts}_best_epoch{epoch+1}"
                    f"_{platform.node()}.pth".replace(":", "-"),
                ),
            )
            print(f"\nNew best result: {val_loss}")
        if not opts.export_at_end and reachedSaveInterval:
            torch.save(
                model.state_dict(),
                os.path.join(
                    results_for_host,
                    f"splinedist_{config.backbone.name}_iter{epoch}.pth",
                ),
            )
            print("Saving a regular interval export")

    print("single-epoch duration:", timeit.default_timer() - start_time)
    print("\n", "-" * 80, "\n", sep="")
if opts.export_at_end:
    torch.save(
        model.state_dict(),
        os.path.join(
            results_for_host,
            f"splinedist_{config.backbone.name}_{config.train_epochs}epochs_"
            + f"{platform.node()}_{train_ts}.pth".replace(":", "-"),
        ),
    )
    plt.savefig(
        os.path.join(
            results_for_host,
            f"splinedist_{config.backbone.name}_{config.train_epochs}epochs_"
            + f"{platform.node()}_{train_ts}.png".replace(":", "-"),
        )
    )

shutil.rmtree(data_dir())
print(f"[Training done] Best result: {current_best}")
os.kill(os.getpid(), signal.SIGTERM)
