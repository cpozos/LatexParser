from os.path import join
from torchvision import transforms
import torch
from PIL import Image

class Data(object):
    IMAGES_DIR = 'src\data\sets\\raw\images'
    LATEX_FORMULAS_PATH = 'src\data\sets\\raw\im2latex_formulas.norm.lst'
    IMAGE_LATEX_DIC_PATH = "src\data\sets\\raw\im2latex_{}_filter.lst"
    OUTPUT_DIR = "src\data\sets\processed"

    """
    docstring
    """
    def __init__(self):
        self._check_data_processed()

        if (not self.is_all_data_processed):
            self._map_latex_formulas()
        pass

    def _check_data_processed(self):
        self.is_train_data_processed = True
        self.is_test_data_processed = True
        self.is_validate_data_processed = True
        self.is_all_data_processed = True

        try:
            f = open(join(Data.OUTPUT_DIR, "train.pkl"))
        except IOError:
            self.is_train_data_processed = False

        try:
            f = open(join(Data.OUTPUT_DIR, "test.pkl"))
        except IOError:
            self.is_test_data_processed = False

        try:
            f = open(join(Data.OUTPUT_DIR, "validate.pkl"))
        except IOError:
            self.is_validate_data_processed = False

        self.is_all_data_processed = self.is_train_data_processed and self.is_test_data_processed and self.is_validate_data_processed
    
    def _map_latex_formulas(self):
        """
        docstring
        """
        # Reads the formulas
        with open(Data.LATEX_FORMULAS_PATH, 'r') as latex_formulas_file:
            self._latex_formulas = [formula.strip('\n') for formula in latex_formulas_file.readlines()]

    def _is_data_processed(self, kind):
        if kind == "train":
            return self.is_train_data_processed
        elif kind == "validation":
            return self.is_validation_data_processed
        elif kind == "test":
            return self.is_test_data_processed
        else:
             return False

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
        if self._is_data_processed(kind):
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

    def get_formula(self, formula_id):
        return self._latex_formulas[formula_id]
    

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

    def get_input_data(self):
        # TODO X data
        return {}


    def get_target_data(self):
        # TODO Y data
        return []