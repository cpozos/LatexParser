from os.path import join
from torchvision import transforms
from PIL import Image
from collections import Counter

import pickle as pkl
import torch
from data.providers._vocabulary import Vocabulary

class Data(object):
    KINDS = ["train", "validation", "test"]
    IMAGES_DIR = 'src\data\sets\\raw\images'
    LATEX_FORMULAS_PATH = 'src\data\sets\\raw\im2latex_formulas.norm.lst'
    IMAGE_LATEX_DIC_PATH = "src\data\sets\\raw\im2latex_{}_filter.lst"
    OUTPUT_DIR = "src\data\sets\processed"

    """
    docstring
    """
    def __init__(self):
        # Images - Latex mappings
        self._check_data_processed()
        if (not self.is_all_data_processed):
            self._process_latex_formulas()
        
        # Vocabulary mapping
        self._process_vocabulary()

    def _check_data_processed(self):
        self.is_all_data_processed = True

        kinds = ["train", "validation", "test"]
        self.kinds_created = dict((kind, True) for kind in Data.KINDS)
        for k in kinds:
            try:
                f = open(join(Data.OUTPUT_DIR, "{}.pkl".format(k)))
            except IOError:
                self.kinds_created[k] = False
            self.is_all_data_processed = self.is_all_data_processed and self.kinds_created[k]

    def _process_latex_formulas(self):
        """
        docstring
        """
        # Reads the formulas
        with open(Data.LATEX_FORMULAS_PATH, 'r') as latex_formulas_file:
            self._latex_formulas = [formula.strip('\n') for formula in latex_formulas_file.readlines()]

    def _process_vocabulary(self, min_count = 10):

        # Checks if vocabulary already created
        self._vocabulary = Vocabulary()
        if not self._vocabulary.is_already_created():
            # Sets the path of the training data
            self._image_latex_data_path = Data.IMAGE_LATEX_DIC_PATH.format('train')

            counter = Counter()
            for pair in self:
                formula = self._latex_formulas[pair[1]].split()
                counter.update(formula)

            for word, count in counter.most_common():
                if count >= min_count:
                    self._vocabulary.add_token(word)

            # Writes processed vocabulary
            self._vocabulary.save()

    def _is_already_created_for(self, kind):
        assert kind in Data.KINDS
        return self.kinds_created[kind]

    def __iter__(self):
        with open(self._image_latex_data_path) as file:
            for line in file:
                #WARNING check which one
                #line = line.strip().split(' ')
                #img_name, formula_id = line[0], line[1]
                img_name, formula_id = line.strip('\n').split()
                img_path = join(Data.IMAGES_DIR, img_name)
                yield img_path, int(formula_id)

    def build_for(self, kind, max = 20):
        # Validates processing
        assert kind in ["train", "validation", "test"]
        if self._is_already_created_for(kind):
            return
        
        # Sets data path
        self._image_latex_data_path = Data.IMAGE_LATEX_DIC_PATH.format(kind)

        # Transforms the raw data
        pairs = []
        transform = transforms.ToTensor()
        for i, pair in enumerate(self):
            if (i > max - 1) :
                break
            img = Image.open(pair[0])
            img_tensor = transform(img)
            formula = self._latex_formulas[pair[1]]
            pair = (img_tensor, formula)
            pairs.append(pair)
            i = i + 1

         # TODO: Check why is sorting
        pairs.sort(key = lambda pair : tuple(pair[0].size()) )

        # Saves processed data
        #TODO not saving for testing
        #out_file = join(Data.OUTPUT_DIR, "{}.pkl".format(kind))
        #torch.save(pairs, out_file)

    # VOCABULARY
    def get_vocabulary(self):
        return self._vocabulary

    # LATEX FORMULAS
    def get_latex_formulas(self):
        return self._latex_formulas

    def get_formula(self, formula_id):
        return self._latex_formulas[formula_id]
    
    def get_input_data(self):
        # TODO X data
        return {}

    def get_target_data(self):
        return []

    #TODO delete it. Deprecated
    def map_images_latex_dictionary(self, kind, max = 100):
        """
        docstring
        max : only to test
        """

        # TODO : a lot of memory use, remove max = 100
        # Reads the Image - LatexFormula dictionary
        pairs = []
        transform = transforms.ToTensor()
        image_latex_dic_path = Data.IMAGE_LATEX_DIC_PATH.format(kind)
        i = 0
        with open(image_latex_dic_path, 'r') as file:
            for line in file:

                if (i > max - 1) :
                     break

                img_name, formula_id = line.strip('\n').split()
                img_path = join(Data.IMAGES_DIR, img_name)
                img = Image.open(img_path)
                img_tensor = transform(img)
                pair = (img_tensor, formula_id)
                pairs.append(pair)
                i = i + 1
            
        # TODO: Check why is sorting
        pairs.sort(key = lambda pair : tuple(pair[0].size()) )
        return pairs

        # TODO Y data
        return []