import matplotlib.pyplot as plt
import torch

def print_tensor(pytorch_tensor):
    plt.imshow(pytorch_tensor.permute(1,2,0) )
    plt.show()


def print_images(dataset, max = 10, range = [0,10]):
    i = 0
    for _tuple in dataset:
        if i > max:
            break
        print_image(_tuple[0])
        i = i + 1
