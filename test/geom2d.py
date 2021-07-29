from splinedist.utils import fill_label_holes
import tifffile
import matplotlib.pyplot as plt

from splinedist.geometry.geom2d import spline_dist

mask = tifffile.imread(r"P:\Robert\splineDist\data\egg\masks\9_11_2020_IMG_0008_5.tif")

mask = fill_label_holes(mask)

sd_res = spline_dist(mask)

print('result from spline_dist?', sd_res)
plt.figure()
plt.imshow(sd_res)
plt.show()
