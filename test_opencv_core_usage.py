from splinedist.rotation import rotate_image
import cv2
import numpy as np

cv2.setNumThreads(8)

testArr = np.random.rand(100, 100, 3)
while True:
    rot_angle = np.random.uniform(-180, 180)
    newArr = rotate_image(testArr, rot_angle)[0]