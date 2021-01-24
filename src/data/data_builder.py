from os.path import join
from torchvision import transforms
from collections import Counter

import pickle as pkl
import torch

# Project
from data.vocabulary import Vocabulary
from data.img_latex_dataset import ImageLatexDataset
from utilities.system import get_system_path

class DataBuilder(object):
    KINDS = ["train", "validate", "test"]
    LATEX_FORMULAS_PATH = 'src\\data\\sets\\raw\\im2latex_formulas.norm.lst'
    IMAGE_LATEX_DIC_PATH = "src\\data\\sets\\raw\\im2latex_{}_filter.lst"
    IMAGES_DIR = 'src\\data\\sets\\raw\\images'  

    """
    docstring
    """
    def __init__(self):
        # Fix paths
        DataBuilder.LATEX_FORMULAS_PATH = get_system_path(DataBuilder.LATEX_FORMULAS_PATH)
        DataBuilder.IMAGE_LATEX_DIC_PATH = get_system_path(DataBuilder.IMAGE_LATEX_DIC_PATH)
        DataBuilder.IMAGES_DIR = get_system_path(DataBuilder.IMAGES_DIR)

        # List of formulas
        self._process_latex_formulas()
        
        # Vocabulary mapping
        self._process_vocabulary()
        

    def __iter__(self):
        with open(self._image_latex_data_path) as file:
            for line in file:
                #WARNING check which one
                #line = line.strip().split(' ')
                #img_name, formula_id = line[0], line[1]
                img_name, formula_id = line.strip('\n').split()
                img_path = join(DataBuilder.IMAGES_DIR, img_name)
                yield img_path, int(formula_id)

    def _process_latex_formulas(self):
        """
        docstring
        """
        # Reads the formulas
        with open(DataBuilder.LATEX_FORMULAS_PATH, 'r') as latex_formulas_file:
            self._latex_formulas = [formula.strip('\n') for formula in latex_formulas_file.readlines()]

    def _process_vocabulary(self, min_count = 10):

        # Checks if vocabulary already created
        self._vocabulary = Vocabulary()
        if not self._vocabulary.is_already_created():
            # Sets the path of the training data
            path = DataBuilder.IMAGE_LATEX_DIC_PATH
            self._image_latex_data_path = path.format('train')

            counter = Counter()
            for pair in self:
                formula_id = pair[1]
                formula = self._latex_formulas[formula_id].split()
                counter.update(formula)

            for word, count in counter.most_common():
                if count >= min_count:
                    self._vocabulary.add_token(word)

            # Writes processed vocabulary
            self._vocabulary.save()

    def get_dataset_for(self, kind, force=False, max_count=None):
        # Validates processing
        assert kind in DataBuilder.KINDS
        
        # Sets data path
        path = DataBuilder.IMAGE_LATEX_DIC_PATH
        path = path.format(kind)

        # Assigns the data set 
        dataset = ImageLatexDataset(kind, max_count=max_count, force=force)

        #TODO check if next logic could be inside ImageLatexDataset
        if not dataset.is_processed_and_saved():

            # Sets the path of the training data to iterate
            self._image_latex_data_path = DataBuilder.IMAGE_LATEX_DIC_PATH.format(kind)
            for pair in self:
                formula_id = pair[1]
                formula = self._latex_formulas[formula_id]
                dataset.add_item(pair[0], formula)           
            dataset.save()
        return dataset

    # VOCABULARY
    def get_vocabulary(self):
        return self._vocabulary

    # LATEX FORMULAS
    def get_latex_formulas(self):
        return self._latex_formulas

    def get_formula(self, formula_id):
        return self._latex_formulas[formula_id]