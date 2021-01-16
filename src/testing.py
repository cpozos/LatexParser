from models.configs import *
from models import *
# Script to test the implementations


# 1. Get data

# 2. Process to train data

# 3. Assign parameters and data to model and create it

# Configurations
e_config = EncoderConfig(20)
d_config = DecoderConfig(50)

model_config = ModelConfig(e_config, d_config, 10, 2)

# Creates the model
model = Model(model_config)

# 4. Train model

# 5. Validate model