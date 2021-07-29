import argparse
import cv2
from glob import glob
import numpy as np

p = argparse.ArgumentParser(description='Resize every image in the specified directory.')

p.add_argument('dir', help='Directory containing images to resize.')
opts = p.parse_args()

imgs = []
for ext in ('png', 'jpg', 'tif'):
    imgs += glob(opts.dir + '/*.%s'%ext)

for img_name in imgs:
    print('Converting image', img_name)
    # img = cv2.imdecode(np.fromfile(img_name, np.uint8), cv2.IMREAD_UNCHANGED)
    print('image shape:', img.shape)
    to_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resized = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite(img_name, to_grayscale)
