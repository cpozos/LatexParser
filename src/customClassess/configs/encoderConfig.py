class EncoderConfig():
    OP_DIM = 512
    DIM = 256

    """
    docstring
    """
    def __init__(self, batch_size, max_height = None, max_width = None):
        self.max_height = 20 if max_height is None else max_height
        self.max_width = 50 if max_width is None else max_width
        self.batch_size = batch_size

    def build(self):
        ## Use PyTorch to build the encoder