#PytTorch
import torch
import torch.nn as nn

class Decoder():
    """
    docstring
    """
    def __init__(self, config):
        self.config = config
        
    def build(self):
        return nn.LSTMCell(
            self.config.dec_rnn_h + self.config.emb_size, 
            self.config.dec_rnn_h)