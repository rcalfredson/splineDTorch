import cv2 as cv
import numpy as np

# img = cv.medianBlur(
#     cv.imread(r"P:\Robert\amazon_annots\images\challenging\2021-7-28_IMG_0004.JPG", 0),
#     5,
# )
img = cv.imread(r"P:\Robert\amazon_annots\images\challenging\2021-7-28_IMG_0004.JPG")
img = cv.medianBlur(img, 5)
img = cv.resize(
    cv.cvtColor(
        img,
        cv.COLOR_BGR2GRAY,
    ),
    (0, 0),
    fx=0.25,
    fy=0.25,
).astype(np.uint8)
print("initial shape of ")
print("shape of image:", img.shape)
circles = cv.HoughCircles(
    img, cv.HOUGH_GRADIENT, 1, 140, param1=35, param2=20, minRadius=30, maxRadius=50
)
circles = np.uint16(np.around(circles))
cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
for i in circles[0, :]:
    # draw the outer circle
    cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
cv.imshow("detected circles (from script)", cimg)
cv.waitKey(0)
cv.destroyAllWindows()
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# mask = (img > 0).astype(np.uint8)
# print('about to erode')
# print(mask)
# eroded = cv.erode(mask, mask)

# img = cv.medianBlur(img,5)
# print(img.dtype)
# th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
# cv.THRESH_BINARY,3,2)

# cv.imshow('debug', cv.resize( th3, (0, 0), fx=0.25, fy=0.25))
# cv.waitKey(0)