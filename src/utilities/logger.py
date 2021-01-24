import time

class TrainingLogger(object):
    def __init__(self, print_freq=1):
        self.print_freq = print_freq
        print("+++++++++++++++ Training initialized +++++++++++++++")        
        self.start = time.time()

    def log_train_step(self, epoch, epochs, step, train_loader_len, step_losses_mean):
        if step % self.print_freq == 0:
            print(f"[Train] Epoch {epoch}/{epochs} Step {step}/{train_loader_len} Loss {step_losses_mean:.6f}")

    def log_val_step(self, epoch, epochs, step_losses_mean):
        print(f"[Valid] Epoch {epoch}/{epochs} Loss {step_losses_mean:.4f}")

    def log_epoch(self, epoch, epochs, training_losses_mean, valid_losses_mean):
        print(f"[Epoch finished] Epoch {epoch}/{epochs} TrainLoss {training_losses_mean:.6f} ValidLoss {valid_losses_mean:.6f} Time {time.time()-self.start} [s]")

    def __del__(self):
        print("+++++++++++++++ Training finished +++++++++++++++")


class TestDataLogger(object):
    def __init__(self, data):
        self._data = data

    def print(self):
        for d in self._data:
            print("===========================================================\n")
            print(f"{d[0]}\n")
            print("-----------------------------------------------------------\n")
            print(f"{d[1]}\n")
            print("===========================================================\n")