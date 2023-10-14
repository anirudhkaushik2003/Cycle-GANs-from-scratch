import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, apply_norm=True):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride, padding=1, padding_mode="replicate")
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

        self.conv1 = Block(img_ch, 64, apply_norm=False) # 128
        self.conv2 = Block(64, 128) # 64
        self.conv3 = Block(128, 256) # 32

        # self.conv4 = Block(256, 512) # 16
        self.conv5 = Block(256, 512, stride=1) # 15
        self.out = nn.Conv2d(512, 1, 4, padding=1) # 15

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        x = self.conv5(x)
        x = self.out(x)

        return x
