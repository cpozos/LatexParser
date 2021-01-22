import time

class TrainingLogger(object):
    def __init__(self, print_freq=1):
        self.print_freq = print_freq
        print("+++++++++++++++ Training initialized +++++++++++++++")        
        self.start = time.time()


    def log_train_step(self, epoch, step, train_loader_len, step_losses_mean):
        if step % self.print_freq == 0:
            print(f"[Train] Epoch {epoch} Step {step}/{train_loader_len} Loss {step_losses_mean:.4f}")

    def log_val_step(self, epoch, step_losses_mean):
        print(f"[Valid] Epoch {epoch} Loss {step_losses_mean:.4f}")

    def log_epoch(self, epoch, training_losses_mean, valid_losses_mean):
        print(f"[Epoch finished] Epoch {epoch} TrainLoss {training_losses_mean:.4f} ValidLoss {valid_losses_mean:.4f} Time {time.time()-self.start}")

    def __del__(self):
        print("+++++++++++++++ Training finished +++++++++++++++")