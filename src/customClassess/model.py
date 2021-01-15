class model(object):
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

    def create(config):


