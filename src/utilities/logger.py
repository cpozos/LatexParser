import statistics
import time

class TrainingLogger(object):
    def __init__(self, print_freq=1):
        print("+++++++++++++++ Training initialized +++++++++++++++")        
        self.start = time.time()

    def log_train_step(epoch, step, train_loader, step_losses):
        if step % print_freq == 0:
            print(f"[Train] Epoch {epoch} Step {step}/{len(train_loader)} Loss {statistics.mean(step_losses):.4f}")

    def log_val_step(epoch, step_losses):
        print(f"[Valid] Epoch {epoch} Loss {statistics.mean(step_losses):.4f}")

    def log_epoc(epoch, training_losses, valid_losses):
        print(f"[Epoch finished] Epoch {epoch} TrainLoss {statistics.mean(training_losses):.4f} ValidLoss {statistics.mean(valid_losses):.4f} Time {time.time()-start}")

    def __del__(self):
        print("+++++++++++++++ Training finished +++++++++++++++")