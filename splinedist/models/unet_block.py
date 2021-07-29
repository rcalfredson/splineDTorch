""" Full assembly of the parts to form the complete network

Copied from this source:
https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
"""


import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # logits = self.outc(x)
        return x

class UNetFromTF(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetFromTF, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.down1 = Down(32, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.middle1 = DoubleConv(128, 256)
        self.middle2 = DoubleConv(256, 128)
        self.up1 = Up(128, 128, skip_conv=True)
        self.post_up1 = SingleConv(192, 64)
        self.up2 = Up(64, 64, skip_conv=True)
        self.post_up2 = SingleConv(96, 32)
        self.up3 = Up(32, 32, skip_conv=True)
        self.post_up3 = SingleConv(64, 32)

    def forward(self, x):
        x_orig = x
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.middle1(x3)
        x5 = self.middle2(x4)
        x = self.up1(x5, x2)
        x = self.post_up1(x)
        x = self.up2(x, x1)
        x = self.post_up2(x)
        x = self.up3(x, x_orig)
        x = self.post_up3(x)
        # logits = self.outc(x)
        return x