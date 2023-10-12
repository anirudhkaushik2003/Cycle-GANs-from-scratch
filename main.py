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
from datasets import HorseDataset
from datasets import ZebraDataset
import random
import itertools

from utils import *

BATCH_SIZE = 1
IMAGE_SIZE = 256

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# torch.use_deterministic_algorithms(True) # Needed for reproducible results

data_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

TRAIN_DIR = "/ssd_scratch/cvit/anirudhkaushik/datasets/cyclegan/horse2zebra/horse2zebra"
VAL_DIR = "/ssd_scratch/cvit/anirudhkaushik/datasets/cyclegan/horse2zebra/horse2zebra"

dataset = HorseDataset(
    root_horse=TRAIN_DIR + "/trainA",
    transform=data_transforms,
)
val_dataset = HorseDataset(
    root_horse=VAL_DIR + "/testA",
    transform=data_transforms,
)

dataset2 = ZebraDataset(
    root_zebra=TRAIN_DIR + "/trainB",
    transform=data_transforms,
)
val_dataset2 = ZebraDataset(
    root_zebra=VAL_DIR + "/testB",
    transform=data_transforms,
)

dataset = torch.utils.data.ConcatDataset([dataset, val_dataset])
dataset2 = torch.utils.data.ConcatDataset([dataset2, val_dataset2])

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
dataloader2 = DataLoader(dataset2, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modelG_1 = Generator(IMAGE_SIZE, 3)
modelD_1 = Discriminator(3)

modelG_1.apply(weights_init)
modelD_1.apply(weights_init)

dummy_img = np.ones((BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))
output_shape = modelD_1(torch.FloatTensor(dummy_img)).shape

modelG_1 = torch.nn.DataParallel(modelG_1)
modelD_1 = torch.nn.DataParallel(modelD_1)


modelG_1 = modelG_1.to(device)
modelD_1 = modelD_1.to(device)

modelG_2 = Generator(256, 3)
modelD_2 = Discriminator(3)

modelG_2.apply(weights_init)
modelD_2.apply(weights_init)

modelG_2 = torch.nn.DataParallel(modelG_2)
modelD_2 = torch.nn.DataParallel(modelD_2)

modelG_2 = modelG_2.to(device)
modelD_2 = modelD_2.to(device)

learning_rate = 1e-5
# optimizerD_1 = torch.optim.Adam(modelD_1.parameters(), lr = learning_rate/10, betas=(0.5, 0.999))
# optimizerD_2 = torch.optim.Adam(modelD_2.parameters(), lr = learning_rate/10, betas=(0.5, 0.999))

optimizerD = torch.optim.Adam(itertools.chain(modelD_1.parameters(), modelD_2.parameters()), lr=learning_rate, betas=(0.5, 0.999))


# optimizerG_1 = torch.optim.Adam(modelG_1.parameters(), lr=learning_rate, betas=(0.5, 0.999))
# optimizerG_2 = torch.optim.Adam(modelG_2.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(itertools.chain(modelG_1.parameters(), modelG_2.parameters()), lr=learning_rate, betas=(0.5, 0.999))

epochs = 100

# G1 takes from normal to monet
# G2 takes from monet to normal

# real_1 = normal
# real_2 = monet

criterion1 = nn.MSELoss() # adversarial loss
criterion2 = nn.L1Loss() # identity loss
criterion3 = nn.L1Loss() # cycle loss (forward)
criterion4 = nn.L1Loss() # cycle loss (backward)

# D1 takes a monet painting
# D2 takes a normal image
def generate_real_images( n_samples, patch_shape):
    y = np.ones((n_samples, 1, patch_shape, patch_shape))
    y = torch.FloatTensor(y)
    return  y

def generate_fake_images(model, dataset, patch_shape ):
    X = model(dataset)
    y = np.zeros((len(X), 1, patch_shape, patch_shape, ))
    y = torch.FloatTensor(y)
    return X, y


def update_image_pool(pool, images, max_size=50):
    selected = list()
    for image in images:
        image = image.cpu().detach().numpy()
        if len(pool) < max_size:
            pool.append(image)
            selected.append(image)
        elif np.random.random() < 0.5:
            selected.append(image)
        else:
            ix = np.random.randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image

    return np.asarray(selected)


################# TRAINING #####################
    
lambda_1 = 5
lambda_2 = 10

real1 = dataloader
real2 = dataloader2

n_epoch, n_batch = epochs, 1
n_patch = output_shape[-1]

pool1, pool2 = list(), list()
batch_per_epoch = int(len(dataset) / BATCH_SIZE)

n_steps = batch_per_epoch * n_epoch


def set_model_grad(model, flag=True, multiGPU=True):
    if multiGPU:
        for param in model.module.parameters():
            param.requires_grad = flag
    else:
        for param in model.parameters():
            param.requires_grad = flag


def train_composite_model(modelG, modelG_2, modelD_1, dataset1, dataset2, y_fake1):
    # adversarial loss
    outputg_1 = modelG(dataset1)
    outputd_1 = modelD_1(outputg_1.detach())
    D_1 = outputd_1.mean().item()
    loss1 = criterion1(outputd_1, y_fake1)
    # identity loss
    outputg_1_id = modelG(dataset2) # give monet to monet generator
    loss2 = criterion2(outputg_1_id, dataset2)
    # cycle loss (forward)  
    outputg_2 = modelG_2(outputg_1) # give generated monet to normal generator
    loss3 = criterion3(outputg_2, dataset1) # reverse the monet to original normal image
    # cycle loss (backward)
    outputg_2_id = modelG_2(dataset1) # give real image to real image generator
    outputg_1_b = modelG(outputg_2_id.detach()) # give identity preserved real image to monet generator
    loss4 = criterion4(outputg_1_b, outputg_1_id.detach())

    lossG = loss1 + lambda_1*loss2 + lambda_2*loss3 + lambda_2*loss4

    return D_1, lossG


for epoch in range(epochs):
    
    for i, (X_real1, X_real2) in enumerate(zip(real1, real2)):
        

        X_real1 = X_real1.to(device)
        X_real2 = X_real2.to(device)

        y_real1 = generate_real_images(BATCH_SIZE, n_patch)
        y_real2 = generate_real_images(BATCH_SIZE, n_patch)

        y_real1 = y_real1.to(device)
        y_real2 = y_real2.to(device)

        X_fake1, y_fake1 = generate_fake_images(modelG_1, X_real2, n_patch)
        X_fake2, y_fake2 = generate_fake_images(modelG_2, X_real1, n_patch)


        X_fake1 = X_fake1.to(device)
        X_fake2 = X_fake2.to(device)
        y_fake1 = y_fake1.to(device)   
        y_fake2 = y_fake2.to(device)

        # update images via buffer
        X_fake1 = update_image_pool(pool1, X_fake1)
        X_fake2 = update_image_pool(pool2, X_fake2)

        if(X_fake1.shape[0] > 1):
            print(X_fake1.shape)

        X_fake1 = torch.FloatTensor(X_fake1)
        X_fake2 = torch.FloatTensor(X_fake2)

        X_fake1 = X_fake1.to(device)
        X_fake2 = X_fake2.to(device)


        set_model_grad(modelD_1, True)
        set_model_grad(modelD_2, True)

        # Train Discriminator1
        # update on real batch 
        optimizerD.zero_grad()
        outputd_1_real = modelD_1(X_real1)
        D_1_real = outputd_1_real.mean().item()
        lossd1_real = criterion1(outputd_1_real, y_real1)
        lossd1_real.backward()

        # Train Discrimator2
        # update on real batch
        outputd2_real = modelD_2(X_real2)
        D_2_real = outputd2_real.mean().item()
        lossd2_real = criterion1(outputd2_real, y_real2)
        lossd2_real.backward()

        optimizerD.step()

        # Train Discriminator1
        # update on fake batch
        outputd_1_fake = modelD_1(X_fake1.detach())
        D_1_fake = outputd_1_fake.mean().item()
        lossd1_fake = criterion1(outputd_1_fake, y_fake1)
        lossd1_fake.backward()

        # Train Discrimator2
        # update on fake batch
        outputd2_fake = modelD_2(X_fake2.detach())
        D_2_fake = outputd2_fake.mean().item()
        lossd2_fake = criterion1(outputd2_fake, y_fake2)
        lossd2_fake.backward()

        optimizerD.step()

        lossd1 = (lossd1_real + lossd1_fake)
        lossd2 = (lossd2_real + lossd2_fake)
        




        
        set_model_grad(modelD_1, False)
        set_model_grad(modelD_2, False)
        #Train Generator (monet -> real) and (real -> monet)
        optimizerG.zero_grad()

        D_1_horse, lossG_1 = train_composite_model(modelG_1, modelG_2, modelD_1, X_real1, X_real2, y_fake1)

        D_2_zebra, lossG_2 = train_composite_model(modelG_2, modelG_1, modelD_2, X_real2, X_real1, y_fake2)

        lossG = lossG_1 + lossG_2
        lossG.backward()
        optimizerG.step()



    if(epoch%1 == 0):
        # print(f"Epoch: {epoch}, LossG_1: {lossG_1}, LossG_2: {lossG_2}, LossD_1: {lossd1}, LossD_2: {lossd2}")
        print(f"Epoch: {epoch:.2f} D_1_real: {D_1_real:.2f}, D_1_fake: {D_1_fake:.2f}, D_2_real: {D_2_real:.2f}, D_2_fake: {D_2_fake:.2f}, D_1_horse: {D_1_horse:.2f}, D_2_zebra: {D_2_zebra:.2f}")
        print(f"loss d1 real: {lossd1_real.item():.2f} loss d1 fake: {lossd1_fake.item():.2f} loss d2 real: {lossd2_real.item():.2f} loss d2 fake: {lossd2_fake.item():.2f}")
    if (epoch)%1 == 0:
        create_checkpoint(modelG_1, optimizerG, epoch, lossG_1, multiGPU=True, type="G1")
        create_checkpoint(modelD_1, optimizerD, epoch, lossd1, multiGPU=True, type="D1")
        create_checkpoint(modelG_2, optimizerG, epoch, lossG_2, multiGPU=True, type="G2")
        create_checkpoint(modelD_2, optimizerD, epoch, lossd2, multiGPU=True, type="D2")
        