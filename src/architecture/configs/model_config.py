class ModelConfig():
    """
    docstring
    """
    def __init__(self, encoderConfig, decoderConfig, cnnConfig, num_layers, num_hidden_layers, epochs):
        self.encoderConfig = encoderConfig
        self.decoderConfig = decoderConfig
        self.cnnConfig = cnnConfig

        # Architecture
        self.num_layers = num_layers
        self.num_hidden_layers = num_hidden_layers
        self.epochs = epochs

    def assign_data(self, data):
        """
        docstring
        """
        self.train_data = data
        return