import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import PIL
import numpy as np
import torchvision
import glob
import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from generator import Generator
from discriminator import Discriminator
from datasets import *
import random
import itertools

from utils import *

BATCH_SIZE = 1
IMAGE_SIZE = 256

multiGPU = False
epochs = 100

lambda_1 = 5
lambda_2 = 10


# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# torch.use_deterministic_algorithms(True) # Needed for reproducible results


# init dataset
data_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])
TRAIN_DIR = "/ssd_scratch/cvit/anirudhkaushik/datasets/cyclegan/apple2orange/apple2orange/"
VAL_DIR = "/ssd_scratch/cvit/anirudhkaushik/datasets/cyclegan/apple2orange/apple2orange/"

dataset = AppleDataset(
    root_apple=TRAIN_DIR + "/trainA",
    transform=data_transforms,
)
val_dataset = AppleDataset(
    root_apple=VAL_DIR + "/testA",
    transform=data_transforms,
)

dataset2 = OrangeDataset(
    root_orange=TRAIN_DIR + "/trainB",
    transform=data_transforms,
)
val_dataset2 = OrangeDataset(
    root_orange=VAL_DIR + "/testB",
    transform=data_transforms,
)

dataset = torch.utils.data.ConcatDataset([dataset, val_dataset])
dataset2 = torch.utils.data.ConcatDataset([dataset2, val_dataset2])

min_length = min(len(dataset), len(dataset2))

#make both datasets of equal length
dataset = torch.utils.data.Subset(dataset, np.arange(min_length))
dataset2 = torch.utils.data.Subset(dataset2, np.arange(min_length))

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
dataloader2 = DataLoader(dataset2, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# initialize models
modelG_1 = Generator(IMAGE_SIZE, 3)
modelD_1 = Discriminator(3)
modelG_1.apply(weights_init)
modelD_1.apply(weights_init)
modelG_2 = Generator(256, 3)
modelD_2 = Discriminator(3)
modelG_2.apply(weights_init)
modelD_2.apply(weights_init)

dummy_img = np.ones((BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))
patch_shape = modelD_1(torch.FloatTensor(dummy_img)).shape[-1]

if multiGPU:
    modelG_1 = torch.nn.DataParallel(modelG_1)
    modelD_1 = torch.nn.DataParallel(modelD_1)
    modelG_2 = torch.nn.DataParallel(modelG_2)
    modelD_2 = torch.nn.DataParallel(modelD_2)

modelG_1 = modelG_1.to(device)
modelG_2 = modelG_2.to(device)
modelD_1 = modelD_1.to(device)
modelD_2 = modelD_2.to(device)

# optimizers
learning_rate = 2e-4
optimizerD = torch.optim.Adam(itertools.chain(modelD_1.parameters(), modelD_2.parameters()), lr=learning_rate, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(itertools.chain(modelG_1.parameters(), modelG_2.parameters()), lr=learning_rate, betas=(0.5, 0.999))

# loss functions
criterionGAN = nn.MSELoss() # adversarial loss
criterionID = nn.L1Loss() # identity loss
criterionCycle = nn.L1Loss() # cycle loss (forward)

### PRINT STATS ###
print("Discriminator Patch shape: ", patch_shape)
print("Number of samples per epoch: ", len(dataset))
print("Multi GPU: ", multiGPU)
print("Learning rate: ", learning_rate)
print("Batch size: ", BATCH_SIZE)
print("Image size: ", IMAGE_SIZE)
print("Number of epochs: ", epochs)
print("Lambda 1: ", lambda_1)
print("Lambda 2: ", lambda_2)
print("Device: ", device)
print("***********************")
print("\n\n\n\n\n\n")
################# TRAINING #####################
    
n_epoch, n_batch = epochs, BATCH_SIZE

pool_zebra, pool_horse = list(), list()
batch_per_epoch = int(len(dataset) / BATCH_SIZE)

n_steps = min_length


def train_composite_model(modelD_Z, modelD_H, fake_zebra, fake_horse, rec_zebra, rec_horse, zebra, horse, ones):
    # adversarial loss
    pred_fake_zebra = modelD_Z(fake_zebra)
    loss_GAN_zebra = criterionGAN(pred_fake_zebra, ones)

    pred_fake_horse = modelD_H(fake_horse)
    loss_GAN_horse = criterionGAN(pred_fake_horse, ones)

    # forward cycle loss
    loss_forward_cycle = criterionCycle(rec_zebra, zebra) * lambda_2
    
    # backward cycle loss
    loss_backward_cycle = criterionCycle(rec_horse, horse) * lambda_2

    lossG = loss_GAN_zebra + loss_GAN_horse + loss_forward_cycle + loss_backward_cycle
    lossG.backward()
    return lossG


def train_D(D, real, fake, ones, zeros):
    # real batch
    pred_real = D(real)
    loss_D_real = criterionGAN(pred_real, ones)
    # fake batch
    pred_fake = D(fake.detach())
    loss_D_fake = criterionGAN(pred_fake, zeros)

    loss_D = (loss_D_real + loss_D_fake)*0.5
    loss_D.backward()

    return loss_D, pred_fake


for epoch in range(epochs):
    lossG_list = []
    lossD_list = []

    D_zebra = 0
    D_horse = 0

    for step, (horse, zebra) in enumerate(zip(dataloader, dataloader2)):

        ones = torch.ones((int(horse.shape[0]), 1, patch_shape, patch_shape)).to(device)
        zeros = torch.zeros((int(horse.shape[0]), 1, patch_shape, patch_shape)).to(device)
        
        horse = horse.to(device)
        zebra = zebra.to(device)

        fake_zebra = modelG_1(horse)
        fake_horse = modelG_2(zebra)
        rec_zebra = modelG_1(fake_horse)
        rec_horse = modelG_2(fake_zebra)

        #  Ds require no gradients when optimizing Gs
        set_model_grad(modelD_1, False)  
        set_model_grad(modelD_2, False)

        # Train Gs
        optimizerG.zero_grad()

        loss_G = train_composite_model(modelD_1, modelD_2, fake_zebra, fake_horse, rec_zebra, rec_horse, zebra, horse, ones)
        # add to lost list
        lossG_list.append(loss_G.item())
        optimizerG.step()


        # set Ds to require gradients
        set_model_grad(modelD_1, True)
        set_model_grad(modelD_2, True)

        # Train Ds
        optimizerD.zero_grad()

        fake_zebra_pool = torch.FloatTensor(update_image_pool(pool_zebra, fake_zebra)).to(device)
        fake_horse_pool = torch.FloatTensor(update_image_pool(pool_horse, fake_horse)).to(device)
        # zebra D
        loss_D_1, pred_zebra = train_D(modelD_1, zebra, fake_zebra_pool, ones, zeros)
        loss_D_2, pred_horse = train_D(modelD_2, horse, fake_horse_pool, ones, zeros)

        loss_D = (loss_D_1 + loss_D_2)*0.5
        lossD_list.append(loss_D.item())

        D_zebra += pred_zebra.mean().item()
        D_horse += pred_horse.mean().item()

        optimizerD.step()

        if (step+1) % 500 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{n_steps}], lossD: {np.mean(lossD_list):.2f}, lossG: {np.mean(lossG_list):.2f}")
            print(f"pred_zebra: {(D_zebra/(step+1)):.2f}, pred_horse: {(D_horse/(step+1)):.2f}")
            print()


    create_checkpoint(modelG_1, epoch, multiGPU, "G1")
    create_checkpoint(modelG_2, epoch, multiGPU, "G2")
    create_checkpoint(modelD_1, epoch, multiGPU, "D1")
    create_checkpoint(modelD_2, epoch, multiGPU, "D2")
        


