# PROJECT
from architecture.configs import ModelConfig
from architecture import *
from data import DataBuilder
#from functools import partial

# TORCH
from torch.utils.data import DataLoader
import torch.optim as optim

# Script to test the implementations

# 0. Preprocess the raw data (only one time) 
data_builder = DataBuilder()
vocabulary = data_builder.get_vocabulary()
dic = vocabulary.token_id_dic
dic = vocabulary.id_token_dic

# 1. Get processed data
data_builder.build_for('train', True)
train_dataset = data_builder.get_dataset()
value = train_dataset [60000] # it contains the tensor 

loader = DataLoader (
    train_dataset,
    batch_size=20,
    #TODO how collate works?
    #collate_fn= partial(collate_fn, voca.token_id_dic),
    pin_memory=False,
    num_workers=4)

# 2. Creates model and optimizer?

model_config = ModelConfig(
    out_size = vocabulary.__len__()
)
model = Model(model_config)
optimizer = optim.Adam(model.parameters(), lr = 3e-4)

# 3. Optimizer


# 4. Train model
# 5. Validate model