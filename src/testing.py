from architecture.configs import *
from architecture import *
from data.providers.data import Data
from data.providers.vocabulary import Vocabulary

# Script to test the implementations

# 0. Preprocess the raw data (only one time) 
data = Data()
list_formulas = data.build_for('train')

voca = Vocabulary()
dic = voca.get_tokens_dic()
dic = voca.get_indexes_dic()

# 1. Get processed data

X = data.get_input_data()
Y = data.get_target_data()

# 2. Process the data to be trained


# 3. Assign parameters and data to model and create it

# Configurations
e_config = EncoderConfig(20)
d_config = DecoderConfig(50)
model_config = ModelConfig(e_config, d_config, 10, 2)

# Creates the model
model = Model(model_config)

# 4. Train model

# 5. Validate model