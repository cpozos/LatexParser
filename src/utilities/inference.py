import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from io import BytesIO

from utilities.tensor import show_tensor_as_image
from utilities.persistance import load_test_data
from utilities.system import apply_system_format, join_paths

class Inference(object):
    def __init__(self, results_path, imgs_path, test_dataset):
        self.imgs_path = apply_system_format(imgs_path)
        self._test_dataset = test_dataset
        self._transform = transforms.ToTensor()
        data = load_test_data(results_path)
        self.targets = data[0]
        self.predictions = data[1]

    def get_inference_data(self, index):
        target = self.targets[index]
        prediction = self.predictions[index]
        img_tensor = self._test_dataset.get_image_tensor(target)
        return img_tensor, target, prediction
        
    def save_latex_as_img(self, latex, img_name):
        fig = plt.figure(figsize=(0.01,0.01))
        fig.text(0, 0, u'${}$'.format(latex),fontsize=12)
        buffer = BytesIO()
        fig.savefig(buffer, dpi=300, transparent=False, format='png', bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)
        buffer.seek(0)

        img_path = join_paths(self.imgs_path, f'pred_img_{img_name}.png')
        with open(img_path, 'wb') as img_file:
            img_file.write(buffer.getvalue())
        
        buffer.close()
        return img_path

    def print_inference_data(self, index):
        tgt_img_tensor, tgt_latex, pred_latex = self.get_inference_data(index)

        # Print target
        print('='*40+' TARGET '+'='*40)
        show_tensor_as_image(tgt_img_tensor)
        print('')
        print(tgt_latex)
        print('')

        # Print prediction
        print('='*40+' PREDICTION '+'='*40)

        # Convert latex to img tensor
        pred_img_name = str(index)
        try:
            pred_img_path = self.save_latex_as_img(pred_latex, pred_img_name)
            pred_img_tensor = self._transform(Image.open(pred_img_path))
            show_tensor_as_image(pred_img_tensor)
        except Exception:
            print("Image could not be generated. Invalid LateX code")
        
        print('')
        print(pred_latex)
        print('')