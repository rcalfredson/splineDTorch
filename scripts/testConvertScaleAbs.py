import cv2

img_in = cv2.imread('/home/tracking/Downloads/2021_7_8_img_0004_nocontrastadjust.png')
cv2.imshow('before', img_in)
img_adj = cv2.convertScaleAbs(img_in, alpha=1, beta=40)
cv2.imshow('after', img_adj)
cv2.waitKey(0)
