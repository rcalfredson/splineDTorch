from glob import glob
import numpy as np
import os
import shutil

base_dir = "/media/Synology3/Robert/splineDist/data/egg"
image_dir = f"{base_dir}/images"
mask_dir = f"{base_dir}/masks"

img_names = glob(f"{image_dir}/*.tif")
heldout_prob = 0.15

for name in img_names:
    if np.random.random() < heldout_prob:
        basename = os.path.basename(name)
        shutil.copy(name, os.path.join(base_dir, "0.15_subset", "images", basename))
        shutil.copy(
            os.path.join(mask_dir, basename),
            os.path.join(base_dir, "0.15_subset", "masks", basename),
        )
