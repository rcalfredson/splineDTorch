import cv2
import numpy as np

mask_path = r"P:\Robert\splineDist\data\egg\masks\9_13_2020_IMG_0001_7.tif"
img_path = r"P:\Robert\splineDist\data\egg\images\9_13_2020_IMG_0001_7.tif"

img = cv2.resize(cv2.imread(img_path), (0, 0), fx=4, fy=4)
mask = cv2.resize(cv2.imread(mask_path), (0, 0), fx=4, fy=4)
mask[np.any(mask != 0, axis = 2)] = (255, 0, 0)
print('mask:', mask)

overlaid = cv2.addWeighted(img, 0.9, mask, 0.3, 0)
cv2.imshow('debug', overlaid)
cv2.waitKey(0)
