class ModelConfig():
    """
    docstring
    """
    def __init__(self, encoderConfig, decoderConfig, num_layers, num_hidden_layers):
        self.encoderConfig = encoderConfig
        self.decoderConfig = decoderConfig
        self.num_layers = num_layers
        self.num_hidden_layers = num_hidden_layers