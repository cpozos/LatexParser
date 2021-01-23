class ModelConfig():
    """
    docstring
    """
    def __init__(self, out_size, emb_size = 0, dec_rnn_h=512, enc_out_dim=512, dropout = 0.):
        """
            out_size : Output size
            emb_size : ??
            dec_rnn_h: ??
            dropout : ??

        """
        # Architecture hyper parameters
        self.out_size = out_size
        self.emb_size = emb_size
        self.dec_rnn_h = dec_rnn_h
        self.enc_out_dim = enc_out_dim
        self.dropout = dropout
        return
