import argparse
import cv2
from glob import glob
import numpy as np
import os
from PIL import Image
import random, string
import sys

sys.path.append(os.path.abspath("./"))
sys.path.append(os.path.abspath("../counting-3"))
from circleFinder import CircleFinder
from splinedist.config import Config
from splinedist.models.database import SplineDistData2D
from splinedist.augmenter import Augmenter


def randID(N=5):
    """Generate uppercase string of alphanumeric characters of length N."""
    return "".join(
        random.SystemRandom().choice(string.ascii_uppercase + string.digits)
        for _ in range(N)
    )


p = argparse.ArgumentParser(
    description="generate image patches using the same "
    "code that creates training/eval patches during experiments"
)
p.add_argument("image_dir", help="directory of images to sample from")
p.add_argument("dest_dir", help="directory to save patches in")
p.add_argument("--config", default="configs/defaults.json")
p.add_argument(
    "--img_rescale_factor",
    type=float,
    default=1.0,
    help="factor by which to scale the images before sampling",
)
p.add_argument(
    "--patch_size",
    type=int,
    nargs='+',
    help="side length of square patches to sample",
)
p.add_argument(
    '--sample_from_whole_imgs',
    action='store_true',
    help="sample from the whole image, i.e., don't segment it into regions first"
)
p.add_argument(
    "--regions_only",
    action="store_true",
    help="save entire egg-laying regions instead of sampling patches from them",
)
opts = p.parse_args()
if opts.patch_size is None:
    opts.patch_size = (160, 160)
else: opts.patch_size = tuple(opts.patch_size)
config = Config(opts.config)
X_names = []
for ext in ("jpg", "JPG"):
    print("searching this glob path:", f"{opts.image_dir}/.{ext}")
    X_names.extend(glob(f"{opts.image_dir}/*.{ext}"))

X_names = random.sample(X_names, 10)

X = list([np.array(img) for img in map(Image.open, X_names)])
if opts.img_rescale_factor != 1.0:
    for i, img in enumerate(X):
        X[i] = cv2.resize(
            img, (0, 0), fx=opts.img_rescale_factor, fy=opts.img_rescale_factor
        )
X_regions = []
X_regions_names = []
print("x:", X)
for i, img in enumerate(X):
    print(f"Finding circles for {X_names[i]}")
    if opts.sample_from_whole_imgs:
        X_regions.append(img)
        X_regions_names.append(X_names[i])
    else:
        try:
            cf = CircleFinder(img, X_names[i], allowSkew=True)
            (
                circles,
                avgDists,
                numRowsCols,
                rotatedImg,
                rotation_angle,
            ) = cf.findCircles(debug=False)
            subimgs = cf.getSubImages(rotatedImg, circles, avgDists, numRowsCols)[0]
            X_regions_names.extend([X_names[i]] * len(subimgs))
            X_regions.extend(subimgs)
        except (KeyError, ValueError):
            continue

Y = [[np.zeros(img.shape[:2])] for img in X_regions]

if opts.regions_only:
    for region in X_regions:
        basename = os.path.basename(X_regions_names[i])
        name_split = os.path.splitext(basename)
        cv2.imwrite(
            os.path.join(opts.dest_dir, f"{name_split[0]}_{randID()}.png"),
            cv2.cvtColor(region, cv2.COLOR_BGR2RGB),
        )
    exit()


augmenter = Augmenter(config, opts, axis_norm=(0, 1), normalize=False)
print("Creating the sampler.")
sampler = SplineDistData2D(
    X_regions,
    Y,
    X_regions_names,
    batch_size=1,
    patch_size=opts.patch_size,
    n_params=None,
    length=len(X_regions_names),
    contoursize_max=None,
    augmenter=augmenter.augment,
    skip_dist_prob_calc=True,
)

for i, img in enumerate(sampler):
    # cv2.imshow(f"debug {X_regions_names[i]}", img[0][0])
    print("name printed:", f"debug {X_regions_names[i]}")
    print("img:", img[0][0])
    basename = os.path.basename(X_regions_names[i])
    name_split = os.path.splitext(basename)
    cv2.imwrite(
        os.path.join(opts.dest_dir, f"{name_split[0]}_{randID()}.png"), img[0][0]
    )
    # cv2.waitKey(0)
