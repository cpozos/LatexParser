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

from utilities.dataloaders import *
from utilities.training import *
from utilities.latex_gen import *

from utilities.tensor import *
from utilities.persistance import *
from utilities.logger import *

def run():

    # Hardware
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_workers = 2

    # ********************************************************************
    # **********************  Hyper parameters  **************************
    # ********************************************************************
    
    # data
    num_data_train = 100000
    num_data_val = 20000
    num_data_test = 2000
    batch_size = 50
    
    # training
    epochs = 10
    learning_rate = 3e-4 #default 3e-4
    drop_out = 0.2 # default 0.2
    clip = 2 #default 2

    # ********************************************************************
    # **********************  Get data  **********************************
    # ********************************************************************
    data_builder = DataBuilder()

    vocabulary = data_builder.get_vocabulary()

    force = True
    train_dataset = data_builder.get_dataset_for('train', max_count=num_data_train, force=force)
    valid_dataset = data_builder.get_dataset_for('validate', max_count=num_data_val, force=force)
    test_dataset = data_builder.get_dataset_for('test', max_count=num_data_test, force=force)

    num_data_train = len(train_dataset)
    num_data_val = len(valid_dataset)
    num_data_test = len(test_dataset)
    print(num_data_train)
    print(num_data_val)
    print(num_data_test)
    
    # Visualize processed data
    #randoms = [train_dataset[random.randint(0, len(train_dataset))][0] for i in range(0,4)]
    #for t in randoms:
    #    print(t.shape)
    #    show_tensor_as_image(t)

    
    # ********************************************************************
    # **********************  Architecture  ******************************
    # ********************************************************************

    model = Model(
        out_size=len(vocabulary),
        enc_out_dim=512,
        emb_size=80,
        dec_rnn_h=512,
        dropout=drop_out
    )

    # ********************************************************************
    # **********************  Training  **********************************
    # ********************************************************************

    # Hyper parameters for training 
    init_epoch = 1

    # For epsilon calculation
    decay_k = 1 #default
    sample_method = "inv_sigmoid" #default ["exp", "inv_sigmoid", "teacher_forcing")

    # Dataloaders
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
        min_lr=3e-5) # default 3e-5

    # Variables to save results
    training_losses = []
    valid_losses = []
    total_step = 0
    best_valid_loss = 1e18
    
    # For profiling
    logger = TrainingLogger(print_freq=10)

    for epoch in range(epochs):
        step_losses = []
        step = 1

        # Training
        model.train()
        loader_len = len(train_loader)
        for imgs_batch, tgt4training_batch, tgt4loss_batch in train_loader:
            optimizer.zero_grad()

            # Epsilon
            #epsilon = cal_epsilon(decay_k, total_step, sample_method)

            # Prediction
            logits = model(imgs_batch, tgt4training_batch, 1.)

            # Compute Loss
            step_loss = cal_loss(logits, tgt4loss_batch)
            
            # Add loss
            step_losses.append(step_loss.item())

            # Print results
            logger.log_train_step(epoch+1, epochs, step, loader_len, statistics.mean(step_losses))

            # Updates
            step_loss.backward()
            clip_grad_norm_(model.parameters(),clip)
            optimizer.step()

            step += 1
            total_step += 1
        
        training_losses.append(statistics.mean(step_losses))

        # Validation
        model.eval()
        step_losses = []
        with torch.no_grad(): # This disable any gradient calculation (better performance)
            for imgs_batch, tgt4training, tgt4loss_batch in valid_loader:

                # Epsilon
                #epsilon = cal_epsilon(decay_k, total_step, sample_method)

                # Prediction
                pred = model(imgs_batch, tgt4training, 1.)

                # Compute loss
                step_loss = cal_loss(pred, tgt4loss_batch)
                step_losses.append(step_loss.item()) 

                # Print results
                logger.log_val_step(epoch+1, epochs, statistics.mean(step_losses))

        # Best validation loss
        valid_loss = statistics.mean(step_losses)
        if valid_loss < best_valid_loss: #best valid loss
            best_valid_loss = valid_loss
            save_model("best_ckpt", model)

        # Scheduler
        lr_scheduler.step(valid_loss)
        valid_losses.append(valid_loss)

        # Save model checkpoint ckpt-e{epoch}
        save_model(f"ckpt-e{epoch+1}-vl{valid_loss:.4f}", model)

        # Print results
        logger.log_epoch(epoch+1, epochs, statistics.mean(training_losses), statistics.mean(valid_losses))

    del logger

    # ********************************************************************
    # **********************  Testing  ***********************************
    # ********************************************************************

    latex_generator = LatexGenerator(model, vocabulary)

    # Loader
    test_loader = DataLoader(
        dataset=test_dataset,
        collate_fn=partial(collate_fn_batch_size_one, vocabulary.token_id_dic)
    )

    # Save testing data
    targets = []
    predictions = []
    for img, formula in test_loader:
        try:
            prediction = latex_generator(img)[0]
            targets.append(formula)
            predictions.append(prediction)
        except RuntimeError:
            break
    
    save_test_data(f"res_{epochs}_{batch_size}_{int(num_data_train/1000)}k_{int(num_data_val/1000)}k_{int(num_data_test/1000)}k", targets, predictions)

if __name__ == '__main__':
    run()