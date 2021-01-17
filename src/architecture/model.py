from architecture import Encoder
from architecture import Decoder

#Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.distributions.uniform import Uniform


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        # Encoder
        self.cnn.encoder = Encoder(self.config.encoder_config).build()

        # Decoder
        self.rnn_decoder = Decoder(self.config.decoder_config).build()

        # Embedding
        self.embedding = nn.Embedding(config.out_size, config.emb_size)

    def create(config):



        return

    def run(self):
        """
        docstring
        """
        for epoch in range(self.config.epochs):
            print ("Ep**Ep")
            iteration = 1

        return


