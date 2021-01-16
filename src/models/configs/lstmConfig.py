class LSTMConfig:
    """
    docstring
    """
    def __init__(self, input_size, num_hidden, num_layers, dropout, use_attention, input_feed,
        use_lookup, vocab_size, batch_size, max_encoder_l, model ):
        self.input_size = input_size
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        self.input_feed = input_feed
        self.use_lookup = use_lookup
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.max_encoder_l = max_encoder_l
        self.model = model