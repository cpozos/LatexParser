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
        

        return

    def map_data():
        """
        """
        map_latex_formulas()

        for kind in ["train", "validate", "test"]:
            
            list_images = map_images_latex_dictionary(kind)

            # Write results
            out_file = join(Data.OUTPUT_DIR, "{}.pkl".format(kind))
            torch.save(list_images, out_file)

        return

    def map_latex_formulas(self):
        """
        docstring
        """
        # Reads the formulas
        with open(Data.LATEX_FORMULAS_PATH, 'r') as latex_formulas_file:
            self.latex_formulas = [formula.strip('\n') for formula in latex_formulas_file.readlines()]
        return self.map_latex_formulas

    def map_images_latex_dictionary(self, kind, max = 100):
        """
        docstring
        max : only to test
        """

        # TODO : a lot of memory use
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