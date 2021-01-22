# Dependencies 
import random
import statistics
from functools import partial
import time

# TORCH
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

# PROJECT
from architecture import *
from data import DataBuilder
from utilities.tensor import *
from utilities.training import *
from utilities.persistance import *

def run():

    # GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 0. Preprocess the raw data (only one time) 
    data_builder = DataBuilder()
    vocabulary = data_builder.get_vocabulary()
    dic = vocabulary.token_id_dic
    dic = vocabulary.id_token_dic

    # 1. Get processed data
    data_builder.build_for('train')
    train_dataset = data_builder.get_dataset()
    data_builder.build_for('validation')
    valid_dataset = data_builder.get_dataset()

    # 1.1 Visualize processed data
    randoms = [train_dataset[random.randint(0, len(train_dataset))][0] for i in range(0,4)]
    #for t in randoms:
    #    print(t.shape)
    #    show_tensor_as_image(t)


    # 2. Create Loaders
    train_loader = DataLoader (
        train_dataset,
        batch_size=50,
        #TODO how collate works?
        # https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278
        collate_fn= partial(collate_fn, vocabulary.token_id_dic),
        pin_memory=False, # It must be False (no GPU): https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
        shuffle=True,
        num_workers=4
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=40
    )

    # 2. Creates model and optimizer?
    learning_rate = 0.01
    model_config = ModelConfig(out_size = len(vocabulary))
    model = Model(model_config)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    # Creates the scheduler
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=0.5, # float default - Learning rate decay rate
        patience=3, # int default - Learning rate decay patience
        verbose=True,
        min_lr=3e-5) # default


    # ********************************************************************
    # **********************  Train loop *********************************
    # ********************************************************************

    #input ("Press Enter to continue with the training")
    
    # Frequencia to print loss
    print_freq = 20 #each 20 steps

    loss_fn = nn.MSELoss(reduction='mean')
    epochs = 2

    # For epsilon
    init_epoch = 1
    total_step = (init_epoch-1)*len(train_loader)
    decay_k = 1 #default
    sample_method = "teacher_forcing" #default (exp, inv_sigmoid)

    # For losses
    training_losses = []
    valid_losses = []

    # For profiling
    start = time.time()
    print(f"Training initialized: ")

    # Training
    for epoch in range(epochs):

        step_losses = []
        step = 0
        for imgs_batch, tgt4training_batch, tgt4loss_batch in train_loader:
            model.train()

            # Epsilon
            epsilon = cal_epsilon(decay_k, total_step, sample_method)

            # Prediction
            result = model(imgs_batch, tgt4training_batch, epsilon)

            # Compute Loss
            step_loss = cal_loss(result, tgt4loss_batch)
            step_loss.backward()

            # Updates parameters and zeroes gradients
            optimizer.step()
            optimizer.zero_grad()

            # Add loss
            step_losses.append(step_loss.item())

            if step % print_freq == 0:
                print(f"[Train] Epoch {epoch} Step {step} Loss {statistics.mean(step_losses):.4f}")

            step += 1
            total_step += 1
        
        training_loss = statistics.mean(step_losses)
        training_losses.append(training_loss)

        # Validation
        mode.eval()
        step_losses = []
        step = 0
        with torch.no_grad(): # This disable any gradient calculation (better performance)
            for imgs_batch, tgt4training, tgt4loss_batch in valid_loader:

                # Epsilon
                epsilon = cal_epsilon(decay_k, total_step, sample_method)

                # Prediction
                pred = model(imgs_batch, tgt4training, epsilon)

                # Compute loss
                step_loss = cal_loss(pred, tgt4loss_batch)
                step_losses.append(step_loss.item()) 

                print(f"[Valid] Epoch {epoch} Step {step} Loss {statistics.mean(step_losses):.4f}")
                step += 1

        valid_loss = statistics.mean(step_losses)
        if valid_loss < best_valid_loss: #best valid loss
            best_valid_loss = valid_loss
            save_model("best_ckpt", model)

        
        # Scheduler
        lr_scheduler.step(valid_loss)
        valid_losses.append(valid_loss)

        # Save model
        save_model("ckpt-{}-{:.4f}".format(epoch,valid_loss))

        # Print results
        print(f"[Epoch finished] Epoch {epoch} TrainLoss {training_loss:.4f}\t ValidLoss {valid_loss:.4f} Time {time.time()-start}")
        
    
    print(f"Training finished ( {time.time()-start} )")

if __name__ == '__main__':
    run()