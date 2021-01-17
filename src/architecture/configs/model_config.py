class ModelConfig():
    """
    docstring
    """
    def __init__(self, out_size, emb_size, dec_rnn_h, dropout,
            encoder_config, decoder_config, num_layers, num_hidden_layers, epochs):
        """
            out_size : Output size
            emb_size : ??
            dec_rnn_h: ??
            dropout : ??

        """
        # Architecture hyper parameters
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.out_size = out_size

        # Training hyper parameters
        self.num_layers = num_layers
        self.num_hidden_layers = num_hidden_layers
        self.epochs = epochs