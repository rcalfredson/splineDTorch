import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SamePadder(nn.Module):
    def __init__(self, filter_shape):
        super(SamePadder, self).__init__()
        self.filter_shape = filter_shape

    def forward(self, input):
        strides = (None, 1, 1)
        in_height, in_width = input.shape[2:4]
        filter_height, filter_width = self.filter_shape

        if in_height % strides[1] == 0:
            pad_along_height = max(filter_height - strides[1], 0)
        else:
            pad_along_height = max(filter_height - (in_height % strides[1]), 0)
        if in_width % strides[2] == 0:
            pad_along_width = max(filter_width - strides[2], 0)
        else:
            pad_along_width = max(filter_width - (in_width % strides[2]), 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom))


class BlockBuilder:
    """Create convolutional blocks for building neural nets."""

    def conv_block(
        self,
        channels: Tuple[int, int],
        size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        N: int = 1,
    ):
        """
        Create a block with N convolutional layers with ReLU activation function.
        The first layer is IN x OUT, and all others - OUT x OUT.

        Args:
            channels: (IN, OUT) - no. of input and output channels
            size: kernel size (fixed for all convolution in a block)
            stride: stride (fixed for all convolution in a block)
            N: no. of convolutional layers

        Returns:
            A sequential container of N convolutional layers.
        """
        # a single convolution + batch normalization + ReLU block
        def block(in_channels):
            layers = [
                SamePadder(size),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=channels[1],
                    kernel_size=size,
                    stride=stride,
                    bias=False,
                    # padding=(size[0] // 2, size[1] // 2),
                ),
                nn.BatchNorm2d(num_features=channels[1]),
                nn.ReLU(),
            ]
            return nn.Sequential(*layers)

        # create and return a sequential container of convolutional layers
        # input size = channels[0] for first block and channels[1] for all others
        return nn.Sequential(*[block(channels[bool(i)]) for i in range(N)])


class FCRN_A(nn.Module):
    """
    Fully Convolutional Regression Network A

    Ref. W. Xie et al. 'Microscopy Cell Counting with Fully Convolutional
    Regression Networks'
    """

    def __init__(self, N: int = 1, input_filters: int = 3, **kwargs):
        """
        Create FCRN-A model with:

            * fixed kernel size = (3, 3)
            * fixed max pooling kernel size = (2, 2) and upsampling factor = 2
            * no. of filters as defined in an original model:
              input size -> 32 -> 64 -> 128 -> 512 -> 128 -> 64 -> 1

        Args:
            N: no. of convolutional layers per block (see conv_block)
            input_filters: no. of input channels
        """
        super(FCRN_A, self).__init__()
        self.input_filters = input_filters
        self.N = N
        bb = BlockBuilder()
        self.model = nn.Sequential(
            # downsampling
            bb.conv_block(channels=(input_filters, 32), size=(3, 3), N=N),
            nn.MaxPool2d(2),
            bb.conv_block(channels=(32, 64), size=(3, 3), N=N),
            nn.MaxPool2d(2),
            bb.conv_block(channels=(64, 128), size=(3, 3), N=N),
            nn.MaxPool2d(2),
            # "convolutional fully connected"
            bb.conv_block(channels=(128, 512), size=(3, 3), N=N),
            # upsampling
            nn.Upsample(scale_factor=2),
            bb.conv_block(channels=(512, 128), size=(3, 3), N=N),
            nn.Upsample(scale_factor=2),
            bb.conv_block(channels=(128, 64), size=(3, 3), N=N),
            nn.Upsample(scale_factor=2),
            bb.conv_block(channels=(64, 32), size=(3, 3), N=N),
        )

    def forward(self, input: torch.Tensor):
        """Forward pass."""
        return self.model(input)