from architecture import Encoder
from architecture import Decoder

class Model(object):
    """
    ## Contains the model
    ## 1. uses a CNN (arranges the feature in a grid)
    ## 2. A RNN encodes each row
    ## 3. A RNN decodes 
    """
    def __init__(self, config):
        self.config = config
        self.encoder = Encoder(self.config.encoderConfig)
        self.decoder = Decoder(self.config.decoderConfig)
        return

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


