class ModelConfig():
    """
    docstring
    """
    def __init__(self, encoder_config, decoder_config, out_size, emb_size, dec_rnn_h, enc_out_dim, dropout = 0.):
        """
            out_size : Output size
            emb_size : ??
            dec_rnn_h: ??
            dropout : ??

        """
        # Architecture hyper parameters
        self.encoder_config = encoder_config
        self.enc_out_dim = enc_out_dim
        self.decoder_config = decoder_config
        self.out_size = out_size
        self.emb_size = emb_size
        self.dec_rnn_h = dec_rnn_h
        self.dropout = dropout
