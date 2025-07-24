import random, torch, os, numpy as np
import torch.nn as nn
import torchvision
from torchvision.utils import save_image
import config
import copy

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def save_some_examples(gen_H, gen_Z, val_loader, epoch, folder, writer):
    os.makedirs(folder, exist_ok=True)

    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)

    gen_H.eval()
    gen_Z.eval()

    with torch.no_grad():
        fake_y = gen_H(x)
        fake_x = gen_Z(y)

        save_image(fake_y * 0.5 + 0.5, f"{folder}/fake_y_{epoch}.png")
        save_image(fake_x * 0.5 + 0.5, f"{folder}/fake_x_{epoch}.png")
        save_image(y * 0.5 + 0.5, f"{folder}/real_y_{epoch}.png")
        save_image(x * 0.5 + 0.5, f"{folder}/real_x_{epoch}.png")

        img_grid_fake = torchvision.utils.make_grid(fake_y, normalize=True)
        img_grid_real = torchvision.utils.make_grid(y, normalize=True)
        writer.add_image("Generated Images", img_grid_fake, global_step=epoch)
        writer.add_image("Real Images", img_grid_real, global_step=epoch)

    gen_H.train()
    gen_Z.train()