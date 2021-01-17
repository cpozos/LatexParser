import torch
import os

from PIL import Image

from os.path import join
from torchvision import transforms
from torch.utils.data import Dataset


class ImageLatexDataset(Dataset):
    OUTPUT_DIR = "src\\data\\sets\\processed"

    def __init__(self, output_filename, max_len = 300, max_count = 10):
        self.out_data_path = join(ImageLatexDataset.OUTPUT_DIR, output_filename + '.pkl')     
        self._max_len = max_len
        self._max_count = max_count
        self._pairs_sorted = False
        self._transform = transforms.ToTensor()
        self.load()

    def load(self):
        if self.is_processed_and_saved():
            self._pairs = torch.load(self.out_data_path)
            #for i, (img, formula) in enumerate(self._pairs):
                ##TODO why self._max_len?
                ## pair = (img, " ".join(formula.split()[:self._max_len]))
                # pair = (img, " ".join(formula.split()))
            self._pairs_sorted = True
        else:
            self._pairs = []

    def dispose(self):
        self._pairs = []
        self._sort_pairs = False

    def add_item(self, img_path, formula):
        if self._max_count < self.__len__() + 1 :
            return

        img = Image.open(img_path)
        img_tensor = self._transform(img)
        self._pairs.append((img_tensor, formula))
        self._pairs_sorted = False

    def save(self):
        self.sort_pairs()
        # Saves processed data
        #TODO not saving for testing
        #return
        torch.save(self._pairs, self.out_data_path)

    def delete(self):
        if self.is_processed_and_saved():
            os.remove(self.out_data_path)
        self.dispose()

    def is_processed_and_saved(self):
        try:
            file = open(self.out_data_path)
        except IOError:
            return False
        return True

    def sort_pairs(self):
        if not self._pairs_sorted:
            # TODO: Check why is sorting
            self._pairs.sort(key = lambda pair : tuple(pair[0].size()) )
            self._pairs_sorted = True

    def get_list(self):
        self.sort_pairs()
        return self._pairs

    def __getitem__(self, index):
        return self._pairs[index]

    def __len__(self):
        return len(self._pairs)

    # def collate_fn(sign2id, batch):
    #    # filter the pictures that have different weight or height
    #    size = batch[0][0].size()
    #    batch = [img_formula for img_formula in batch
    #            if img_formula[0].size() == size]
    #    # sort by the length of formula
    #    batch.sort(key=lambda img_formula: len(img_formula[1].split()),
    #            reverse=True)

    #    imgs, formulas = zip(*batch)
    #    formulas = [formula.split() for formula in formulas]
    #    # targets for training , begin with START_TOKEN
    #    tgt4training = formulas2tensor(add_start_token(formulas), sign2id)
    #    # targets for calculating loss , end with END_TOKEN
    #    tgt4cal_loss = formulas2tensor(add_end_token(formulas), sign2id)
    #    imgs = torch.stack(imgs, dim=0)
    #    return imgs, tgt4training, tgt4cal_loss

    #def formulas2tensor(formulas, sign2id):
    #    """convert formula to tensor"""

    #    batch_size = len(formulas)
    #    max_len = len(formulas[0])
    #    tensors = torch.ones(batch_size, max_len, dtype=torch.long) * PAD_TOKEN
    #    for i, formula in enumerate(formulas):
    #        for j, sign in enumerate(formula):
    #            tensors[i][j] = sign2id.get(sign, UNK_TOKEN)
    #    return tensors