from elasticdeform import deform_random_grid
import numpy
from scipy.ndimage.interpolation import rotate

rand_img = numpy.empty((400, 200))
points_to_deform = numpy.array([[4, 30],
[200, 100]])
points_to_deform_2 = numpy.array([[4],
[200]])
input_arrays = [rand_img, points_to_deform]
# print('points will turn out to be:', [3] * len(rand_img.shape))
crop_arg = [slice(75, 140), slice(120, 180)]
print(deform_random_grid(input_arrays, order=3, crop=crop_arg, rotate=40, axis=0))