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
    plt.grid(False)
    plt.imshow(pytorch_tensor.permute(1,2,0))
    plt.show()

def tensor2formula(self, tensor, idx2token, pretty=False, tags=True):
    if not pretty:
        if tags:
            return ' '.join(idx2token[tensor[i]] for i in range(tensor.shape[0]))
        else:
            return ' '.join(idx2token[tensor[i]] for i in range(tensor.shape[0])
                            if idx2token[tensor[i]] not in ['<s>', '</s>', '<pad>'])
    else:
        s = ' '.join(idx2token[tensor[i]] for i in range(tensor.shape[0]))
        end = s.find('</s>')
        if end != -1 : end = end - 1
        s = s[4:end]
        s = s.replace('<pad>', '')
        s = s.replace('<unk>', '')
        return s
