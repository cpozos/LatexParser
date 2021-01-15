class DecoderConfig():
    DIM = 512
    
    """
    docstring
    """
    def __init__(self, batch_size, max_height = None, max_width = None):
        self.max_height = 20 if max_height is None else max_height
        self.max_width = 50 if max_width is None else max_width
        self.max_ct_vec_length = self.max_height * self.max_width

    def build(self):
        #Use PyTorch to build the decoder