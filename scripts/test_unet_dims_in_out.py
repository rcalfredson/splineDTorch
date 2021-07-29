import cv2
import os
import sys
sys.path.append(os.path.abspath("./"))
import numpy as np
import torch
import torch.nn as nn
from splinedist.models.unet_block import UNet

test_net = UNet(3, 1, bilinear=False).cuda()
test_img = cv2.imread(r"P:\Robert\objects_counting_dmap\egg_source\combined_robert_uli_temp\2020-11-20_img_0002_1_1_left_IRY6P.jpg")
test_img = np.expand_dims(np.transpose(test_img, (2, 0, 1)), axis=0)
print('Input shape:', test_img.shape)
print(test_img)
input_t = torch.from_numpy(test_img).float().cuda()
# maxpool_test = nn.MaxPool2d(2)
# t_after_maxpool = maxpool_test(input_t)
# print('shape after performing only max pooling:', t_after_maxpool.shape)
result = test_net.forward(input_t)
print('Shape of the result:', result.shape)