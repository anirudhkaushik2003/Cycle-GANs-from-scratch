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

def update_image_pool(pool, images, max_size=50):
    selected = list()
    for image in images:
        image = image.cpu().detach().numpy()
        if len(pool) < max_size:
            pool.append(image)
            selected.append(image)
        elif np.random.random() < 0.5: # 50% chance of using current image
            selected.append(image)
        else:
            ix = np.random.randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image # 50% chance of using one of the stored images

    return np.asarray(selected)

def set_model_grad(model, flag=True, multiGPU=False):
    if multiGPU:
        for param in model.module.parameters():
            param.requires_grad = flag
    else:
        for param in model.parameters():
            param.requires_grad = flag

def create_checkpoint(model, epoch, multiGPU=False, type="G"):
    if not multiGPU:
        filename = f'/ssd_scratch/cvit/anirudhkaushik/checkpoints/cycle_gan{type}_checkpoint_{epoch}_epoch.pt'

        checkpoint = {
            'model': model.state_dict(),
            'epoch': epoch,
        }
        torch.save(checkpoint, filename)

        # save latest
        filename = f'/ssd_scratch/cvit/anirudhkaushik/checkpoints/cycle_gan{type}_checkpoint_latest.pt'
        checkpoint = {
            'model': model.state_dict(),
            'epoch': epoch,
        }
        torch.save(checkpoint, filename)

    else:
        filename = f'/ssd_scratch/cvit/anirudhkaushik/checkpoints/cycle_gan{type}_checkpoint_{epoch}_epoch.pt'
        checkpoint = {
            'model': model.module.state_dict(),
            'epoch': epoch,
        }
        torch.save(checkpoint, filename)

        # save latest
        filename = f'/ssd_scratch/cvit/anirudhkaushik/checkpoints/cycle_gan{type}_checkpoint_latest.pt'
        checkpoint = {
            'model': model.module.state_dict(),
            'epoch': epoch,
        }
        torch.save(checkpoint, filename)


def restart_last_checkpoint(model, optimizer, multiGPU=False, type="G"):
    filename = f'/ssd_scratch/cvit/anirudhkaushik/checkpoints/cycle_gan{type}_checkpoint_latest.pt'
    if not multiGPU:
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        print(f"Restarting from epoch {epoch}")
    else:
        checkpoint = torch.load(filename)
        model.module.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        print(f"Restarting from epoch {epoch}")

    return epoch