from architecture.configs import *
from architecture import *
from data import Data

# Script to test the implementations

# 0. Preprocess the raw data (only one time) 
data = Data()
voca = data.get_vocabulary()
dic = voca.token_id_dic
dic = voca.id_token_dic

# 1. Get processed data
data.build_for('train')
train_dataset = data.get_dataset()

data.build_for('test')
test_dataset = data.get_dataset()

data.build_for('validation')
validation_dataset = data.get_dataset()

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