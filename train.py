"""Main script used to train networks."""
import argparse
import elasticdeform
from splinedist.constants import DEVICE
from matplotlib import pyplot as plt
from numpy.core.defchararray import mod
import torch
from csbdeep.utils import normalize
import datetime
from glob import glob
import json
import numpy as np
import os
from pathlib import Path
import platform
import shutil
import signal
from splinedist.config import Config
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
from tqdm import tqdm

import cv2

# python train.py --config configs\fcrn_backbone.json --plot --left_col_plots scatter --val_interval 2
## temp command: python trainByBatchLinux.py 10 "--config configs/unet_with_deform.json --plot --val_interval 4"
# eval command: python eval_via_pt_data.py '/media/Synology3/Robert/objects_counting_dmap/egg_source/heldout_robert_WT_5' '/media/Synology3/Robert/splineDTorch/saved_models/unet_deform_sigma=6/complete_nets'

sys_type = platform.system()
win_drive_letter = "P"
if sys_type == "Windows" and "Dell2" in platform.node():
    win_drive_letter = "R"
DATA_BASE_DIR = {
    "Windows": f"{win_drive_letter}:/Robert/splineDist/data/egg",
    "Linux": "/media/Synology3/Robert/splineDist/data/egg",
}[sys_type]
train_ts = datetime.datetime.now()

lr_history = {}


def options():
    parser = argparse.ArgumentParser(
        description="Train object detection model using SplineDist"
    )
    parser.add_argument(
        "--config",
        help="Config file to use (default: configs/defaults.json).",
        default="configs/defaults.json",
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
    return parser.parse_args()


opts = options()
if not os.path.isdir(data_dir(must_exist=False)):
    Path(data_dir(must_exist=False)).mkdir(parents=True, exist_ok=True)
X = sorted(glob(os.path.join(DATA_BASE_DIR, "images/*.tif")))
Y = sorted(glob(os.path.join(DATA_BASE_DIR, "masks/*.tif")))
assert all(Path(x).name == Path(y).name for x, y in zip(X, Y))
X = list(map(imread, X))
Y = list(map(imread, Y))
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
axis_norm = (0, 1)  # normalize channels independently

if n_channel > 1:
    print(
        "Normalizing image channels %s."
        % ("jointly" if axis_norm is None or 2 in axis_norm else "independently")
    )
    sys.stdout.flush()

# X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]

assert len(X) > 1, "not enough training data"
rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(0.21 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
# X_val = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X_val)]
X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
print("number of images: %3d" % len(X))
print("- training:       %3d" % len(X_trn))
print("- validation:     %3d" % len(X_val))

i = min(9, len(X) - 1)
img, lbl = X[i], Y[i]
assert img.ndim in (2, 3)
img = img if (img.ndim == 2 or img.shape[-1] == 3) else img[..., 0]

M = 8
n_params = 2 * M
grid = (2, 2)

contoursize_max = get_contoursize_max(Y_trn)
config_kwargs = {"n_channel_in": n_channel, "contoursize_max": contoursize_max}
config = Config(opts.config, **config_kwargs)
if opts.plot:
    plt.ion()
    fig, plots = plt.subplots(nrows=2, ncols=2)
else:
    plots = [None] * 2

phi_generator(M, int(config.contoursize_max))
grid_generator(M, config.train_patch_size, config.grid)


def write_lr_history(config: Config):
    if not hasattr(config, "lr_history_filename"):
        setattr(
            config,
            "lr_history_filename",
            f"splinedist_{config.backbone.name}_lr_history"
            f"_{platform.node()}_{train_ts}.json".replace(":", "-"),
        )
    with open(config.lr_history_filename, "w") as my_f:
        json.dump(lr_history, my_f, ensure_ascii=False, indent=4)


def random_fliprot(img: torch.Tensor, mask: torch.Tensor):
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(perm)
    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    img = normalize(img, 1, 99.8, axis=axis_norm)
    return img, mask


def random_intensity_change(img):
    img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
    return img


def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02 * np.random.uniform(0, 1)
    x = x + sig * np.random.normal(0, 1, x.shape)
    return x, y


model = SplineDist2D(config)
model.cuda()
median_size = calculate_extents(list(Y), np.median)
fov = np.array(model._axes_tile_overlap("YX"))
print(f"median object size:      {median_size}")
print(f"network field of view:   {fov}")
if any(median_size > fov):
    print(
        "WARNING: median object size larger than field of view of the neural network."
    )

optimizer = torch.optim.Adam(model.parameters(), config.train_learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=config.lr_reduct_factor, verbose=True, patience=config.lr_patience
)
model.prepare_for_training()

train_looper = Looper(
    model,
    config,
    DEVICE,
    model.loss,
    optimizer,
    augmenter,
    X_trn,
    Y_trn,
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
    augmenter,
    X_val,
    Y_val,
    validation=True,
    plots=plots[1],
    left_col_plots=opts.left_col_plots,
)

current_best = np.infty
for i, epoch in enumerate(range(config.train_epochs)):
    start_time = timeit.default_timer()
    current_lr = optimizer.param_groups[0]["lr"]
    if len(lr_history) == 0 or lr_history["last"] != current_lr:
        lr_history["last"] = current_lr
        lr_history[f"epoch_{i+1}"] = current_lr
        write_lr_history(config)
    start_time = timeit.default_timer()
    print(f"Epoch {epoch + 1}\n")

    train_looper.run(i)
    if i % opts.val_interval == 0:
        with torch.no_grad():
            val_loss = valid_looper.run(i)
        lr_scheduler.step(val_loss)
        new_best = val_loss < current_best
        reachedSaveInterval = i % 20 == 0
        if new_best:
            current_best = val_loss
            for f in glob(
                f"splinedist_{config.backbone.name}_{train_ts}_best_"
                f"*_{platform.node()}.pth".replace(":", "-")
            ):
                os.unlink(f)
            torch.save(
                model.state_dict(),
                f"splinedist_{config.backbone.name}_{train_ts}_best_epoch{i+1}"
                f"_{platform.node()}.pth".replace(":", "-"),
            )
            print(f"\nNew best result: {val_loss}")
        if not opts.export_at_end and reachedSaveInterval:
            torch.save(
                model.state_dict(),
                f"splinedist_{config.backbone.name}_iter{i}.pth",
            )
            print("Saving a regular interval export")

    print("single-epoch duration:", timeit.default_timer() - start_time)
    print("\n", "-" * 80, "\n", sep="")
if opts.export_at_end:
    torch.save(
        model.state_dict(),
        f"splinedist_{config.backbone.name}_{config.train_epochs}epochs_"
        + f"{platform.node()}_{train_ts}.pth".replace(":", "-"),
    )
    plt.savefig(
        f"splinedist_{config.backbone.name}_{config.train_epochs}epochs_"
        + f"{platform.node()}_{train_ts}.png".replace(":", "-")
    )

shutil.rmtree(data_dir())
print(f"[Training done] Best result: {current_best}")
os.kill(os.getpid(), signal.SIGTERM)
