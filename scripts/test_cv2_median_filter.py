import cv2

mask_to_smooth = cv2.imread(
    r'/home/tracking/Pictures/mask_smoothing_test_img.png'
)
cv2.imshow('original', mask_to_smooth)
smoothed = cv2.medianBlur(mask_to_smooth, 3)
cv2.imshow('smoothed', smoothed)
cv2.waitKey(0)