from os.path import join
from torchvision import transforms
from collections import Counter

import pickle as pkl
import torch

# Project
from data.providers._vocabulary import Vocabulary
from data.providers._img_latex_dataset import ImageLatexDataset
from utilities.system import get_system_path

class DataBuilder(object):
    KINDS = ["train", "validation", "test"]
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
                formula = self._latex_formulas[pair[1]].split()
                counter.update(formula)

            for word, count in counter.most_common():
                if count >= min_count:
                    self._vocabulary.add_token(word)

            # Writes processed vocabulary
            self._vocabulary.save()

    def build_for(self, kind, force = False):
        # Validates processing
        assert kind in DataBuilder.KINDS
        
        # Sets data path
        path = DataBuilder.IMAGE_LATEX_DIC_PATH
        path = path.format(kind)

        # Assigns the data set 
        self._dataset = ImageLatexDataset(kind, force)

        #TODO check if next logic could be inside ImageLatexDataset
        if not self._dataset.is_processed_and_saved():

            # Sets the path of the training data to iterate
            self._image_latex_data_path = DataBuilder.IMAGE_LATEX_DIC_PATH.format('train')
            for pair in self:
                formula = self._latex_formulas[pair[1]]
                self._dataset.add_item(pair[0], formula)              
            self._dataset.save()

    # VOCABULARY
    def get_vocabulary(self):
        return self._vocabulary

    # LATEX FORMULAS
    def get_latex_formulas(self):
        return self._latex_formulas

    def get_formula(self, formula_id):
        return self._latex_formulas[formula_id]
    
    # DATA SETS
    def get_dataset(self):
        return self._dataset


    #TODO delete it. Deprecated
    #def map_images_latex_dictionary(self, kind, max = 100):

        ## TODO : a lot of memory use, remove max = 100
        ## Reads the Image - LatexFormula dictionary
        #pairs = []
        #transform = transforms.ToTensor()
        #image_latex_dic_path = DataBuilder.IMAGE_LATEX_DIC_PATH.format(kind)
        #i = 0
        #with open(image_latex_dic_path, 'r') as file:
        #    for line in file:

        #        if (i > max - 1) :
        #             break

        #        img_name, formula_id = line.strip('\n').split()
        #        img_path = join(DataBuilder.IMAGES_DIR, img_name)
        #        img = Image.open(img_path)
        #        img_tensor = transform(img)
        #        pair = (img_tensor, formula_id)
        #        pairs.append(pair)
        #        i = i + 1
        #    
        ## TODO: Check why is sorting
        #pairs.sort(key = lambda pair : tuple(pair[0].size()) )
        #return pairs

        ## TODO Y data
        #return []