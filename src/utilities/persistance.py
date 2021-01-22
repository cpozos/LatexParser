from utilities.system import get_current_path
import torch

def save_model(self, model_name, model):
    save_path = join(get_current_path(), model_name+'.pt')
    print("Saving checkpoint to {}".format(save_path))

    torch.save(model, save_path)

    #torch.save({
    #    'epoch': self.epoch,
    #    'model_state_dict': self.model.state_dict(),
    #    'optimizer_state_dict': self.optimizer.state_dict(),
    #    'lr_sche': self.lr_scheduler.state_dict(),
    #    'epoch': self.epoch,
    #    'args': self.args
    #}, save_path)