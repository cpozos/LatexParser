# Dependencies 
import random
import numpy as np
from functools import partial
import time

# TORCH
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

# PROJECT
from architecture import *
from data import DataBuilder
from utilities.tensor import *
from utilities.training import *

def run():
    # 0. Preprocess the raw data (only one time) 
    data_builder = DataBuilder()
    vocabulary = data_builder.get_vocabulary()
    dic = vocabulary.token_id_dic
    dic = vocabulary.id_token_dic

    # 1. Get processed data
    data_builder.build_for('train', False)
    train_dataset = data_builder.get_dataset()
    data_builder.build_for('validation')
    valid_dataset = data_builder.get_dataset()

    # 1.1 Visualize processed data
    randoms = [train_dataset[random.randint(0, len(train_dataset))][0] for i in range(0,4)]
    #for t in randoms:
    #    print(t.shape)
    #    show_tensor_as_image(t)


    # 2. Create Loaders
    data_loader = DataLoader (
        train_dataset,
        batch_size=100,
        #TODO how collate works?
        # https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278
        collate_fn= partial(collate_fn, vocabulary.token_id_dic),
        pin_memory=False, # It must be False (no GPU): https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
        num_workers=3
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=100
    )

    # 2. Creates model and optimizer?
    learning_rate = 0.01
    model_config = ModelConfig(out_size = len(vocabulary))
    model = Model(model_config)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)


    # 3. Train loop
    input ("Press Enter to continue with the training")

    #device = 
    loss_fn = nn.MSELoss(reduction='mean')
    epochs = 2
    training_losses = []
    valid_losses = []


    start = time.time()
    print(f"Training initialized: ")
    for epoch in range(epochs):

        # Training
        batch_losses = []

        # 
        for i, data in enumerate(data_loader):
            imgs_batch, tgt4training_batch, tgt4loss_batch  = data
            model.train()

            # Make prediction
            result = model(imgs_batch, tgt4training_batch)

            # Compute Loss
            loss = cal_loss(result, tgt4loss_batch)
            
            # Gradients
            loss.backward()

            # Updates parameters and zeroes gradients
            optimizer.step()
            optimizer.zero_grad()

            # Add loss
            batch_losses.append(loss.item())

            print(f"Train batch step {i}. Batch loss {loss.item()}")
        
        training_loss = np.mean(batch_losses)
        training_losses.append(training_loss)

        # Evaluation
        batch_losses = []
        with torch.no_grad(): # This disable any gradient calculation (better performance)
            for imgs_batch, formulas_batch in valid_loader:
                mode.eval()
                pred = model(imgs_batch, formulas_batch)

                # Compute loss
                batch_loss = loss_fn(formulas_batch, pred)
                batch_losses.append(batch_loss.item()) 

                print(f"Validate batch step. Validate loss {batch_loss.item()}")

        valid_loss = np.mean(batch_losses)
        valid_losses.append(valid_loss)

        print(f"[{epoch+1}] Training loss: {training_loss:.3f}\t Validation loss: {valid_loss:.3f}")
        print(f"Time step ( {time.time()-start} )")


    print(f"Training finished ( {time.time()-start} )")
    torch.save(model,get_current_path())
    # model = torch.load(get_current_path())

if __name__ == '__main__':
    run()