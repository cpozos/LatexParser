import matplotlib.pyplot as plt
import torch

img_count = 0

def describe_tensor(tensor):
    print("Type: {}".format(tensor.type))
    print("Shape/size: {}".format(tensor.shape))
    print("Values: \n{}".format(tensor))

def save_tensor_as_image(pytorch_tensor):
    global img_count
    plt.imshow(pytorch_tensor.permute(1,2,0))
    plt.savefig("{}.png".format(str(img_count)))
    img_count = img_count + 1

def show_tensor_as_image(pytorch_tensor):
    plt.imshow(pytorch_tensor.permute(1,2,0))
    plt.show()
