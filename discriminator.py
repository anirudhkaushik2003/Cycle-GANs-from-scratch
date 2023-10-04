import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, apply_norm=True):
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride, 1)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.relu = nn.LeakyReLU(0.2)
        self.apply_norm = apply_norm

    def forward(self, x):
        x = self.conv(x)
        if self.apply_norm:
            x = self.norm(x)
        x = self.relu(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, img_ch):
        super(Discriminator, self).__init__()
        self.img_ch = img_ch

        # splitting images into 70x70 patches is done implicitly, calculate receptive field in reverse order to confirm

        self.conv1 = Block(img_ch, 64, apply_norm=False)
        self.conv2 = Block(64, 128)
        self.conv3 = Block(128, 256)

        self.conv4 = Block(256, 512, 4, stride=1)
        self.out = nn.Conv2d(512, 1, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.out(x)
        x = self.sigmoid(x)
