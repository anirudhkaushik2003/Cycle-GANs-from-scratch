import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import PIL


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_ch, out_ch, stride=stride, kernel_size=3, padding="same"
        )
        self.norm1 = nn.InstanceNorm2d(out_ch)
        self.relu = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(
            out_ch, out_ch, stride=stride, kernel_size=3, padding="same"
        )
        self.norm2 = nn.InstanceNorm2d(out_ch)

    def forward(self, x):
        x_cp = x.clone()
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)

        # x = torch.cat((x, x_cp), dim=1)
        x = x + x_cp
        return x


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=4, s=2, p=1):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=k, stride=s, padding=p
        )
        self.norm = nn.InstanceNorm2d(out_ch)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x
    
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=4, s=2, p=1):
        super(UpBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=k, stride=s, padding=p, output_padding=1
        )
        self.norm = nn.InstanceNorm2d(out_ch)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class Generator(nn.Module):
    def __init__(self, img_size, img_channels):
        super(Generator, self).__init__()
        self.img_channels = img_channels
        self.img_size = img_size

        self.conv1 = DownBlock(img_channels, 64, 7, 1, p=3) # conv2d

        self.conv2 = DownBlock(64, 128, 3, 2, 1) # down sample
        self.conv3 = DownBlock(128, 256, 3, 2, 1) # down sample
        
        res_blocks = []

        for _ in range(9):
            res_blocks.append(ResBlock(256, 256))

        self.res_blocks = nn.Sequential(*res_blocks)

        self.conv4 = UpBlock(256, 128, 3, 2, 1) # up sample
        self.conv5 = UpBlock(128, 64, 3, 2, 1) # up sample

        self.conv6 = DownBlock(64, img_channels, 7, 1, p=3) # conv2d
        self.tanh = nn.Tanh()

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res_blocks(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.tanh(x)
        return x