import cv2

loadedImg = cv2.imread(
    r"P:\Robert\splineDist\data\arena_pit\masks\8_5_2020_IMG_0003.tif"
)
print(loadedImg)
cv2.imshow('debug', loadedImg)
cv2.waitKey(0)
print('image dtype:', loadedImg.dtype)
print('image shape:', loadedImg.shape)
contours, _ = cv2.findContours(loadedImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

print('Done.', contours)