import torch
import os

from PIL import Image

from os.path import join
from torchvision import transforms
from torch.utils.data import Dataset

# Project
from utilities.system import get_system_path


class ImageLatexDataset(Dataset):
    OUTPUT_DIR = "src\\data\\sets\\processed"

    def __init__(self, output_filename, max_count=None, force=False, max_len=300):
        # Fix paths
        ImageLatexDataset.OUTPUT_DIR = get_system_path(ImageLatexDataset.OUTPUT_DIR)
        self.max_form_len = 0

        self.out_data_path = join(ImageLatexDataset.OUTPUT_DIR, output_filename + '.pkl')  
        self._max_count = max_count   
        self._max_len = max_len
        self._pairs_sorted = False
        self._transform = transforms.ToTensor()

        if force:
            self.delete()
        
        self._load()

    def _load(self):
        """
        Tries to load persistent data if it exists.
        Otherwise, it sets _pairs to empty
        """
        self._pairs = []
        if self.is_processed_and_saved():
            self._pairs = torch.load(self.out_data_path)
            self._set_max_formulas_len()
            #for i, (img, formula) in enumerate(self._pairs):
                ##TODO why self._max_len?
                ## pair = (img, " ".join(formula.split()[:self._max_len]))
                # pair = (img, " ".join(formula.split()))
            

    def dispose(self):
        self._pairs = []
        self._pairs_sorted = False

    def add_item(self, img_path, formula):
        self._pairs.append((img_path, formula))
        self._pairs_sorted = False

    def save(self):
        """
        Saves persistent data
        """
        # Saves processed data
        #TODO not saving for testing
        #return
        torch.save(self._pairs, self.out_data_path)
        self._set_max_formulas_len()
        
    def delete(self):
        """
        Deletes persistent data if exists
        """
        if self.is_processed_and_saved():
            os.remove(self.out_data_path)
        self.dispose()

    def refresh(self):
        self.delete()
        self._load()

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

    def _set_max_formulas_len(self):
        self.max_form_len = max([len(pair[1]) for pair in self._pairs])

    def __getitem__(self, index):
        img_path, formula = self._pairs[index]
        img_tensor = self._transform(Image.open(img_path))
        return (img_tensor, formula)

    def get_image_tensor(self, formula):
        for pair in self._pairs:
            if formula.strip() == pair[1].strip():
                img_path = pair[0]
                img_tensor = self._transform(Image.open(img_path))
                return img_tensor
        return None
    
    def __len__(self):
        count = len(self._pairs)
        if self._max_count is not None and self._max_count <= count:
            return self._max_count
        else:
            return count 

    def print(self):
        print(f"Formulas: {self.__len__()}")
        print(f"Max formula length: {self.max_form_len}")
