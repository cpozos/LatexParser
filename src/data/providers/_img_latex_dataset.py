import torch
import os

from PIL import Image

from os.path import join
from torchvision import transforms

class ImageLatexDataset(object):
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
            for i, (img, formula) in enumerate(pairs):
                #TODO why self._max_len?
                pair = (img, " ".join(formula.split()[:self._max_len]))
        else:
            self._pairs = []

    def dispose(self):
        self._pairs = []
        self._sort_pairs = False

    def add_item(self, img_path, formula):
        if self._max_count < len(self._pairs) + 1 :
            return

        img = Image.open(img_path)
        img_tensor = self._transform(img)
        self._pairs.append((img_tensor, formula))
        self._pairs_sorted = False

    def save(self):
        self._sort_pairs()
        # Saves processed data
        #TODO not saving for testing
        return
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

    def _sort_pairs(self):
        if not self._pairs_sorted:
            # TODO: Check why is sorting
            self._pairs.sort(key = lambda pair : tuple(pair[0].size()) )
            self._pairs_sorted = True

    def get_data_set(self):
        self._sort_pairs()
        return self._pairs