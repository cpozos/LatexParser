class lstm(object):
    """
    docstring
    """
    
    def create(self, config):
        self.input_size = config.input_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_layers = config.num_layers
        self.dropout = 