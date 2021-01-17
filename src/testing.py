from architecture.configs import *
from architecture import *
from data import Data
#from functools import partial

from torch.utils.data import DataLoader

# Script to test the implementations

# 0. Preprocess the raw data (only one time) 
data = Data()
voca = data.get_vocabulary()
dic = voca.token_id_dic
dic = voca.id_token_dic

# 1. Get processed data
data.build_for('train')
train_dataset = data.get_dataset()

loader = DataLoader (
    data.get_dataset(),
    batch_size=20,
    #TODO how collate works?
    #collate_fn= partial(collate_fn, voca.token_id_dic),
    pin_memory=False,
    num_workers=4)

# 2. Assign parameters and data to model and create it

# Configurations
e_config = EncoderConfig(20)
d_config = DecoderConfig(50)
model_config = ModelConfig(e_config, d_config, 10, 2)

# Creates the model
model = Model(model_config)

# 4. Train model

# 5. Validate model