import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import PIL

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Block, self).__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.relu = nn.LeakyReLU(0.2)

class Generator(nn.Module):
    def __init__(self, img_size, img_channels):
        super(Generator, self).__init__()
        self.img_channels = img_channels
        self.img_size = img_size    

        
