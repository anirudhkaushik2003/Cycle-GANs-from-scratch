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
from datasets import HorseZebraDataset

from utils import *

BATCH_SIZE = 1
IMAGE_SIZE = 256

data_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

TRAIN_DIR = "/ssd_scratch/cvit/anirudhkaushik/datasets/cyclegan/horse2zebra/horse2zebra"
VAL_DIR = "/ssd_scratch/cvit/anirudhkaushik/datasets/cyclegan/horse2zebra/horse2zebra"

dataset = HorseZebraDataset(
    root_horse=TRAIN_DIR + "/trainA",
    root_zebra=TRAIN_DIR + "/trainB",
    transform=transforms,
)
val_dataset = HorseZebraDataset(
    root_horse=VAL_DIR + "/testA",
    root_zebra=VAL_DIR + "/testB",
    transform=transforms,
)

# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
# dataloader2 = DataLoader(dataset2, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modelG_1 = Generator(256, 3*2)
modelD_1 = Discriminator(3*2)

modelG_1.apply(weights_init)
modelD_1.apply(weights_init)

dummy_img = np.ones((BATCH_SIZE, 6, IMAGE_SIZE, IMAGE_SIZE))
output_shape = modelD_1(torch.FloatTensor(dummy_img)).shape
print(output_shape)
exit(0)
modelG_1 = torch.nn.DataParallel(modelG_1)
modelD_1 = torch.nn.DataParallel(modelD_1)


modelG_1 = modelG_1.to(device)
modelD_1 = modelD_1.to(device)

modelG_2 = Generator(256, 3*2)
modelD_2 = Discriminator(3*2)

modelG_2.apply(weights_init)
modelD_2.apply(weights_init)

modelG_2 = torch.nn.DataParallel(modelG_2)
modelD_2 = torch.nn.DataParallel(modelD_2)

modelG_2 = modelG_2.to(device)
modelD_2 = modelD_2.to(device)

learning_rate = 2e-4
optimizerD_1 = torch.optim.Adam(modelD_1.parameters(), lr = learning_rate, betas=(0.5, 0.999))
optimizerD_2 = torch.optim.Adam(modelD_2.parameters(), lr = learning_rate, betas=(0.5, 0.999))

optimizerG_1 = torch.optim.Adam(modelG_1.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizerG_2 = torch.optim.Adam(modelG_2.parameters(), lr=learning_rate, betas=(0.5, 0.999))

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
def generate_real_images(dataset, n_samples, patch_shape):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return X, y

def generate_fake_images(model, dataset, patch_shape ):
    X = model(dataset)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


def update_image_pool(pool, images, max_size=50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            pool.append(image)
            selected.append(image)
        elif np.random.random() < 0.5:
            selected.append(image)
        else:
            ix = np.random.randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
            selected.append(pool[ix])
            pool[ix] = image

    return np.asarray(selected)


################# TRAINING #####################
    

real1 = dataset
real2 = dataset2

n_epoch, n_batch = epochs, 1
n_patch = output_shape[-1]

pool1, pool2 = list(), list()
batch_per_epoch = int(len(dataset) / BATCH_SIZE)

n_steps = batch_per_epoch * n_epoch

for i in range(n_steps):
    X_real1, y_real1 = generate_real_images(real1, BATCH_SIZE, n_patch)
    X_real2, y_real2 = generate_real_images(real2, BATCH_SIZE, n_patch)

    X_fake1, y_fake1 = generate_fake_images(modelG_1, X_real2, n_patch)
    X_fake2, y_fake2 = generate_fake_images(modelG_2, X_real1, n_patch)

    # update images via buffer
    X_fake1 = update_image_pool(pool1, X_fake1)
    X_fake2 = update_image_pool(pool2, X_fake2)

    #Train Generator (monet -> real)
    optimizerG_2.zero_grad()

    # adversarial loss
    outputg_2 = modelG_2(real2) # give monet to normal generator
    outputd_2 = modelD_2(outputg_2)
    loss1G2 = criterion1(outputd_2, y_fake2)
    loss1G2.backward()
    # identity loss
    outputg_2_id = modelG_2(real1) # give normal to normal generator
    loss2G2 = criterion2(outputg_2_id, real1)
    loss2G2 = 5*loss2G2
    loss2G2.backward()
    # cycle loss (forward)  
    outputg_1 = modelG_1(outputg_2) # give generated normal to monet generator
    loss3G2 = criterion3(outputg_1, real2) # convert the generated normal to monet image
    loss3G2 = 10*loss3G2
    loss3G2.backward()
    # cycle loss G2(backward)
    outputg_1_id = modelG_1(real2) # give monet image to monet image generator
    outputg_2_b = modelG_2(outputg_1_id) # give identity preserved real image to monet generator
    loss4G2 = criterion4(outputg_2_b, outputg_2_id)
    loss4G2 = 10*loss4G2
    loss4G2.backward()

    lossG_2 = loss1G2 + loss2G2 + loss3G2 + loss4G2

    optimizerG_2.step()

    # Train Discriminator1
    # update on real batch 
    optimizerD_1.zero_grad()
    outputd_1_real = modelD_1(real1)
    lossd1_real = criterion1(outputd_1_real, y_real1)
    lossd1_real = lossd1_real*0.5
    lossd1_real.backward()

    # update on fake batch
    outputd_1_fake = modelD_1(X_fake1.detach())
    lossd1_fake = criterion1(outputd_1_fake, y_fake1)
    lossd1_fake = lossd1_fake*0.5
    lossd1_fake.backward()

    lossd1 = (lossd1_real + lossd1_fake)

    # Train Generator (real -> monet)
    optimizerG_1.zero_grad()

    # adversarial loss
    outputg_1 = modelG_1(real1)
    outputd_1 = modelD_1(outputg_1)
    loss1 = criterion1(outputd_1, y_fake1)
    loss1.backward()
    # identity loss
    outputg_1_id = modelG_1(real2) # give monet to monet generator
    loss2 = criterion2(outputg_1_id, real2)
    loss2 = 5*loss2
    loss2.backward()
    # cycle loss (forward)  
    outputg_2 = modelG_2(outputg_1) # give generated monet to normal generator
    loss3 = criterion3(outputg_2, real1) # reverse the monet to original normal image
    loss3 = 10*loss3
    loss3.backward()
    # cycle loss (backward)
    outputg_2_id = modelG_2(real1) # give real image to real image generator
    outputg_1_b = modelG_1(outputg_2_id) # give identity preserved real image to monet generator
    loss4 = criterion4(outputg_1_b, outputg_1_id)
    loss4 = 10*loss4
    loss4.backward()

    lossG_1 = loss1 + loss2 + loss3 + loss4

    optimizerG_1.step()

    # Train Discrimator2
    # update on real batch
    outputd2_real = modelD_2(real2)
    lossd2_real = criterion1(outputd2_real, y_real2)
    lossd2_real = lossd2_real*0.5
    lossd2_real.backward()

    # update on fake batch
    outputd2_fake = modelD_2(X_fake2.detach())
    lossd2_fake = criterion1(outputd2_fake, y_fake2)
    lossd2_fake = lossd2_fake*0.5
    lossd2_fake.backward()

    lossd2 = lossd2_real + lossd2_fake

    optimizerD_2.step()

    print("Epoch: ", i, "LossD1: ", lossd1, "LossG1: ", lossG_1, "LossD2: ", lossd2, "LossG2: ", lossG_2)




real_labels_monet = torch.ones(real1.size(0), 1).to(device)

# Train Discriminator





