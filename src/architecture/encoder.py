#PytTorch
import torch
import torch.nn as nn

class Encoder(object):
    """
    docstring
    """
    def __init__(self,config):
        self._config = config

    def build(self):
        return nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2,1),

            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2,1),

            nn.Conv2d(128,256,3,1,1),
            nn.ReLU(),

            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(),

            nn.MaxPool2d((2,1),(2,1),0),

            nn.Conv2d(256, self._config.out_size,3,1,0),
            nn.ReLU()
        )