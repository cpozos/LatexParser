#PytTorch
import torch
import torch.nn as nn

class Decoder():
    """
    docstring
    """
    def __init__(self, in_size, hidden_size):
        self._in_size = in_size
        self._hidden_size = hidden_size
        
    def build(self):
        return nn.LSTMCell (self._in_size, self._hidden_size)