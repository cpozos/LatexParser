class EncoderConfig():
    OP_DIM = 512
    DIM = 256

    """
    docstring
    """
    def __init__(self, out_size, out_dim, batch_size, max_height = None, max_width = None):
        self.out_dim = out_dim
        self.out_size = out_size

        # not used
        self.max_height = 20 if max_height is None else max_height
        self.max_width = 50 if max_width is None else max_width
        self.batch_size = batch_size