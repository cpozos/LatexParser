from customClassess.configs import *
from customClassess import *
# Script to test the implementations

# Configurations
e_config = EncoderConfig(20)
d_config = DecoderConfig(50)

model_config = ModelConfig(e_config, d_config, 10, 2)

# Creates the model
model = Model(model_config)