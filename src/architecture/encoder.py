#PytTorch
import torch
import torch.nn as nn

class Encoder(object):
    """
    docstring
    """
    def __init__(self, out_dim):
        self._out_dim = out_dim

    def build(self):
        return nn.Sequential(

            # (BatchSize, NumberChannels, Height, Width)
            # Square kernels 3x3
            nn.Conv2d(in_channels: 3, out_channels: 64, kernel_size: 3, stride: 1, padding: 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size: 2, stride: 2, padding: 1),

            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2,1),

            nn.Conv2d(128,256,3,1,1),
            nn.ReLU(),

            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(),

            nn.MaxPool2d((2,1),(2,1),0),

            nn.Conv2d(256, self._out_dim,3,1,0),
            nn.ReLU()
        )