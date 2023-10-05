import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import PIL
import numpy as np

import glob
import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from generator import Generator
from discriminator import Discriminator

from utils import *

BATCH_SIZE = 1

dataset = []
dataset2 = []
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
dataloader2 = DataLoader(dataset2, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modelG_1 = Generator(256, 3*2)
modelD_1 = Discriminator(256, 3*2)

modelG_1.apply(weights_init)
modelD_1.apply(weights_init)

modelG_1 = torch.nn.DataParallel(modelG_1)
modelD_1 = torch.nn.DataParallel(modelD_1)

modelG_1 = modelG_1.to(device)
modelD_1 = modelD_1.to(device)

modelG_2 = Generator(256, 3*2)
modelD_2 = Discriminator(256, 3*2)

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

for epoch in epochs:
    for ((real1, _), (real2, _)) in list(zip((dataloader, dataloader2))):
        real1 = real1.to(device)
        real2 = real2.to(device)

        real_labels_monet = torch.ones(real1.size(0), 1).to(device)

        # Train Discriminator


        # Train Generator
        optimizerG_1.zero_grad()

        # adversarial loss
        outputg_1 = modelG_1(real1)
        outputd_1 = modelD_1(outputg_1)
        loss1 = criterion1(outputd_1, real_labels_monet)
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
        outputg_2_id = modelG_2(real2) # give real image to real image generator
        outputg_1_b = modelG_1(outputg_2_id) # give identity preserved real image to monet generator
        loss4 = criterion4(outputg_1_b, outputg_1_id)
        loss4 = 10*loss4
        loss4.backward()

        lossG_1 = loss1 + loss2 + loss3 + loss4

        optimizerG_1.step()


