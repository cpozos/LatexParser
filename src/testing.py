import random

# PROJECT
from architecture.configs import ModelConfig
from architecture import *
from data import DataBuilder
from utilities.tensor import *
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
data_builder.build_for('train', False)
train_dataset = data_builder.get_dataset()
value = train_dataset [60000] # it contains the tensor 


randoms = [train_dataset[random.randint(0, len(train_dataset))][0] for i in range(0,100)]

for t in randoms:
    print(t.shape)
    
## Visualize images
# save_tensor_as_image(train_dataset[0][0])
show_tensor_as_image(train_dataset[1][0])

loader = DataLoader (
    train_dataset,
    batch_size=20,
    #TODO how collate works?
    # https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278
    collate_fn= partial(collate_fn, voca.token_id_dic),
    pin_memory=False, # It must be False (no GPU): https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
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