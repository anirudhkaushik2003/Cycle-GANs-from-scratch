import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, datasets
import cv2

IMAGE_SIZE = 256


class ZebraDataset(Dataset):
    def __init__(self, root_zebra, transform=None):
        self.root_zebra = root_zebra
        self.transform = transform

        self.zebra_images = os.listdir(root_zebra)
        self.length_dataset = len(self.zebra_images) # 1000, 1500
        self.zebra_len = len(self.zebra_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        zebra_img = self.zebra_images[index % self.zebra_len]

        zebra_path = os.path.join(self.root_zebra, zebra_img)

        zebra_img = Image.open(zebra_path).convert("RGB")


        if self.transform:
            zebra_img = self.transform(zebra_img)

        return zebra_img
    

class HorseDataset(Dataset):
    def __init__(self, root_horse, transform=None):
        self.root_horse = root_horse
        self.transform = transform

        self.horse_images = os.listdir(root_horse)
        self.length_dataset = len(self.horse_images) # 1000, 1500
        self.horse_len = len(self.horse_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        horse_img = self.horse_images[index % self.horse_len]

        horse_path = os.path.join(self.root_horse, horse_img)

        horse_img = Image.open(horse_path).convert("RGB")

        if self.transform:
            horse_img = self.transform(horse_img)

        return horse_img
    


class AppleDataset(Dataset):
    def __init__(self, root_apple, transform=None):
        self.root_apple = root_apple
        self.transform = transform

        self.apple_images = os.listdir(root_apple)
        self.length_dataset = len(self.apple_images) # 1000, 1500
        self.apple_len = len(self.apple_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        apple_img = self.apple_images[index % self.apple_len]

        apple_path = os.path.join(self.root_apple, apple_img)

        apple_img = Image.open(apple_path).convert("RGB")


        if self.transform:
            apple_img = self.transform(apple_img)

        return apple_img
    

class OrangeDataset(Dataset):
    def __init__(self, root_orange, transform=None):
        self.root_orange = root_orange
        self.transform = transform

        self.orange_images = os.listdir(root_orange)
        self.length_dataset = len(self.orange_images) # 1000, 1500
        self.orange_len = len(self.orange_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        orange_img = self.orange_images[index % self.orange_len]

        orange_path = os.path.join(self.root_orange, orange_img)

        orange_img = Image.open(orange_path).convert("RGB")

        if self.transform:
            orange_img = self.transform(orange_img)

        return orange_img