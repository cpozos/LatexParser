from utilities.system import get_current_path
from utilities.system import join_paths

import torch
import pickle as pkl

def save_model(model_name, model):
    save_path = join_paths(get_current_path(), f"{model_name}.pt")
    print(f"Saving checkpoint to {save_path}")
    torch.save(model, save_path)

    #torch.save({
    #    'epoch': self.epoch,
    #    'model_state_dict': self.model.state_dict(),
    #    'optimizer_state_dict': self.optimizer.state_dict(),
    #    'lr_sche': self.lr_scheduler.state_dict(),
    #    'epoch': self.epoch,
    #    'args': self.args
    #}, save_path)

def save_test_data(file_name,references, results):
    file_path = join_paths(get_current_path(), f"{file_name}.pkl")
    with open(file_path, "wb") as file:
            pkl.dump([references, results], file)

def load_test_data(file_path):
    with open(file_pathgt,"rb") as file:
        data = pkl.load(file)
    return data