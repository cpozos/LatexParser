# Dependencies 
import random
import statistics
from functools import partial
import time
import tqdm

# TORCH
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

# PROJECT
from architecture import *
from data import DataBuilder
from utilities.training import *
from utilities.testing import *

from utilities.tensor import *
from utilities.persistance import *
from utilities.logger import *

def run():
    # HARDWARE
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_workers = 1

    # ********************************************************************
    # **********************  Get data  **********************************
    # ********************************************************************
    data_builder = DataBuilder()

    vocabulary = data_builder.get_vocabulary()

    force = True
    train_dataset = data_builder.get_dataset_for('train', max_count=50, force=force)
    valid_dataset = data_builder.get_dataset_for('validation', max_count=20, force=force)
    test_dataset = data_builder.get_dataset_for('test', max_count=2, force=force)

    # Visualize processed data
    #randoms = [train_dataset[random.randint(0, len(train_dataset))][0] for i in range(0,4)]
    #for t in randoms:
    #    print(t.shape)
    #    show_tensor_as_image(t)

    
    # ********************************************************************
    # **********************  Architecture  ******************************
    # ********************************************************************

    model = Model(out_size=len(vocabulary))

    # ********************************************************************
    # **********************  Training  *********************************
    # ********************************************************************

    #input ("Press Enter to continue with the training")
    
    # Hyper parameters for training 
    init_epoch = 1
    epochs = 1
    learning_rate = 0.01

    # For epsilon calculation
    decay_k = 1 #default
    sample_method = "teacher_forcing" #default ["exp", "inv_sigmoid")

    # Dataloaders
    batch_size = 1
    train_loader = DataLoader (
        train_dataset,
        batch_size=batch_size,
        #TODO how collate works?
        # https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278
        collate_fn=partial(collate_fn, vocabulary.token_id_dic),
        pin_memory=False, # It must be False (no GPU): https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
        #shuffle=True,
        num_workers=num_workers
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, vocabulary.token_id_dic)
    )

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        factor=0.5, # float default - Learning rate decay rate
        patience=3, # int default - Learning rate decay patience
        verbose=True,
        min_lr=3e-5) # default

    # Variables to save results
    total_step = 0
    training_losses = []
    valid_losses = []
    best_valid_loss = 1e18
    
    # For profiling
    logger = TrainingLogger(print_freq=10)

    for epoch in range(epochs):

        step_losses = []
        step = 0

        # Training
        model.train()
        for imgs_batch, tgt4training_batch, tgt4loss_batch in train_loader:
            optimizer.zero_grad()

            # Epsilon
            epsilon = cal_epsilon(decay_k, total_step, sample_method)

            # Prediction
            logits = model(imgs_batch, tgt4training_batch, epsilon)

            # Compute Loss
            step_loss = cal_loss(logits, tgt4loss_batch)
            
            # Updates
            step_loss.backward()
            optimizer.step()

            # Add loss
            step_losses.append(step_loss.item())

            # Print results
            logger.log_train_step(epoch, step, len(train_loader), statistics.mean(step_losses))

            step += 1
            total_step += 1
        
        training_loss = statistics.mean(step_losses)
        training_losses.append(training_loss)

        # Validation
        model.eval()
        step_losses = []
        with torch.no_grad(): # This disable any gradient calculation (better performance)
            for imgs_batch, tgt4training, tgt4loss_batch in valid_loader:

                # Epsilon
                epsilon = cal_epsilon(decay_k, total_step, sample_method)

                # Prediction
                pred = model(imgs_batch, tgt4training, epsilon)

                # Compute loss
                step_loss = cal_loss(pred, tgt4loss_batch)
                step_losses.append(step_loss.item()) 

                # Print results
                logger.log_val_step(epoch, statistics.mean(step_losses))

        # Best validation loss
        valid_loss = statistics.mean(step_losses)
        if valid_loss < best_valid_loss: #best valid loss
            best_valid_loss = valid_loss
            #save_model("best_ckpt", model)

        # Scheduler
        lr_scheduler.step(valid_loss)
        valid_losses.append(valid_loss)

        # Save model checkpoint
        # save_model("ckpt-{}-{:.4f}".format(epoch,valid_loss), model)

        # Print results
        logger.log_epoch(epoch, statistics.mean(training_losses), statistics.mean(valid_losses))

    del logger

    # ********************************************************************
    # **********************  Testing  *******************************
    # ********************************************************************

    gen = LatexGenerator(model, vocabulary)

    # Loader
    test_loader = DataLoader(test_dataset, 
        batch_size=1,
        collate_fn=partial(collate_fn, vocabulary.token_id_dic),
        pin_memory=False,
        num_workers=num_workers
    )
    
    result_file = join_paths(get_current_path(), "resFile.")
    imgs, tgt4training, tgt4loss_batch = next(iter(test_loader))
    ref = gen._idx2formulas(tgt4loss_batch)
    logit = gen(imgs)

    # Testing
    for imgs, tgt4training, tgt4loss_batch in test_loader:
        try:
            reference = gen._idx2formulas(tgt4loss_batch)
            results = gen(imgs)
        except RuntimeError:
            break

    for i in range(len(results)):
        print(f"{reference[i]} {results[i]}")

if __name__ == '__main__':
    run()