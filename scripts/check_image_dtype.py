import cv2
from glob import glob
import numpy as np
import os

image_files = glob('/media/Synology3/Robert/splineDist/data/arena_pit/images/*.tif')
for i, f in enumerate(image_files):
    img = cv2.imread(f)
    print(f'Checking image {i}')
    if img.dtype != np.uint8:
        print(f"found dtype other than float32 for {os.path.basename(f)}")
        print(img.dtype)
        input()