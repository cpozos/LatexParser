from os.path import join
from torchvision import transforms
import torch
from PIL import Image

class Data(object):
    IMAGES_DIR = '..\sets\raw\images'
    LATEX_FORMULAS_PATH = '..\sets\raw\im2latex_formulas.norm.lst'
    IMAGE_LATEX_DIC_PATH = "..\sets\raw\im2latex_{}_filter.lst"
    OUTPUT_DIR = "..\sets\processed"

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
            out_file = join(OUTPUT_DIR, "{}.pkl".format(kind))
            torch.save(list_images, out_file)

        return

    def map_images_latex_dictionary(self, kind, formulas):
        """
        docstring
        """

        # Reads the Image - LatexFormula dictionary
        pairs = []
        transform = transforms.ToTensor()
        image_latex_dic_path = IMAGE_LATEX_DIC_PATH.format(kind)
        with open(filter, 'r') as file:
            for line in file:
                img_name, formula_id = line.strip('\n').split()
                image_path = join(IMAGES_DIR, img_name)
                image = Image.open(img_path)
                img_tensor = transform(img)
                pair = (img_tensor, formula_id)
                pairs.append(pair)
            
        # TODO: Check why is sorting
        pairs.sort(key = lambda pair : tuple(pair[0].size()) )
        return pairs

    def map_latex_formulas(self):
        """
        docstring
        """
        # Reads the formulas
        with open(LATEX_FORMULAS_PATH, 'r') as latex_formulas_file:
            self.latex_formulas = [formula.strip('\n') for formula in latex_formulas_file.readlines()]