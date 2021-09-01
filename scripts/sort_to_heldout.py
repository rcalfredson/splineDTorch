from glob import glob
import numpy as np
import os
import shutil

base_dir = "/media/Synology3/Robert/splineDist/data/arena_pit/patches"
image_dir = f"{base_dir}/images"
mask_dir = f"{base_dir}/masks"

img_names = glob(f"{image_dir}/*.tif")
heldout_prob = .1

for name in img_names:
    if np.random.random() < heldout_prob:
        basename = os.path.basename(name)
        shutil.move(name, os.path.join(
            base_dir,
            'heldout','images',
            basename
        ))
        shutil.move(os.path.join(
            mask_dir,
            basename
        ),
        os.path.join(
            base_dir,
            'heldout', 'masks',
            basename
        ))