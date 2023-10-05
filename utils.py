# custom weights initialization called on ``netG`` and ``netD``
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import PIL

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def create_checkpoint(model, optimizer, epoch, loss, multiGPU=False, type="G"):
    if not multiGPU:
        filename = f'/ssd_scratch/cvit/anirudhkaushik/checkpoints/cycle_gan{type}_checkpoint_{epoch}_epoch.pt'

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss
        }
        torch.save(checkpoint, filename)

        # save latest
        filename = f'/ssd_scratch/cvit/anirudhkaushik/checkpoints/cycle_gan{type}_checkpoint_latest.pt'
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss
        }
        torch.save(checkpoint, filename)

    else:
        filename = f'/ssd_scratch/cvit/anirudhkaushik/checkpoints/cycle_gan{type}_checkpoint_{epoch}_epoch.pt'
        checkpoint = {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss
        }
        torch.save(checkpoint, filename)

        # save latest
        filename = f'/ssd_scratch/cvit/anirudhkaushik/checkpoints/cycle_gan{type}_checkpoint_latest.pt'
        checkpoint = {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss
        }
        torch.save(checkpoint, filename)


def restart_last_checkpoint(model, optimizer, multiGPU=False, type="G"):
    filename = f'/ssd_scratch/cvit/anirudhkaushik/checkpoints/cycle_gan{type}_checkpoint_latest.pt'
    if not multiGPU:
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Restarting from epoch {epoch}")
    else:
        checkpoint = torch.load(filename)
        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Restarting from epoch {epoch}")

    return epoch, loss