#PytTorch
import torch
import torch.nn as nn

class CNNEncoder(object):
    """
    docstring
    """
    def __init__(self, out_dim):
        self._out_dim = out_dim
        self.cnn = nn.Sequential(

            # [BatchSize, NumberChannels, Height, Width]
            # Square kernels 3x3
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # [B, 128, H, W]
            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2,1),

            nn.Conv2d(128,256,3,1,1),
            nn.ReLU(),

            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d((2,1),(2,1),0),

            nn.Conv2d(256, self._out_dim,3,1,0),
            nn.ReLU()

            # [B, 512, H, W]
        )

    def encode(self, imgs):
        """
        Applies the CNN layer to encode the images
        args:
            imgs: tensors of dimension [B,3,H,W]
        """
        encoded_imgs = self.cnn(imgs)  #[B, 512, H', W']
        encoded_imgs = encoded_imgs.permute(0, 2, 3, 1)  #[B, H', W', 512]

        # Unfold the image to get a sequence
        B, H, W, _ = encoded_imgs.shape
        encoded_imgs = encoded_imgs.contiguous().view(B, H*W, -1)
        return encoded_imgs # [B, H' * W']