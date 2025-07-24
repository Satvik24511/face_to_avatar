import torch
from torch.utils.tensorboard import SummaryWriter
from dataset import ImageDataset
from utils import save_checkpoint, load_checkpoint, save_some_examples
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
import os
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator

def train_fn(disc_A, disc_B, gen_A, gen_B, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, writer, epoch):
    loop = tqdm(loader, leave=True)
    A_reals = 0
    A_fakes = 0
    B_reals = 0
    B_fakes = 0
    for idx, (img_A, img_B) in enumerate(loop):
        img_A = img_A.to(config.DEVICE)
        img_B = img_B.to(config.DEVICE)
        
        with torch.amp.autocast(device_type=config.DEVICE):
            fake_B = gen_B(img_A)
            D_B_real = disc_B(img_B)
            D_B_fake = disc_B(fake_B.detach())
            B_reals += D_B_real.mean().item()
            B_fakes += D_B_fake.mean().item()
            D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
            D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))
            D_B_loss = D_B_real_loss + D_B_fake_loss

            fake_A = gen_A(img_B)
            D_A_real = disc_A(img_A)
            D_A_fake = disc_A(fake_A.detach())
            A_reals += D_A_real.mean().item()
            A_fakes += D_A_fake.mean().item()
            D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
            D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
            D_A_loss = D_A_real_loss + D_A_fake_loss

            D_loss = (D_B_loss + D_A_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        with torch.amp.autocast(device_type=config.DEVICE):
            D_B_fake = disc_B(fake_B)
            D_A_fake = disc_A(fake_A)
            loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))
            loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))

            cycle_A = gen_A(fake_B)
            cycle_B = gen_B(fake_A)
            cycle_A_loss = l1(img_A, cycle_A)
            cycle_B_loss = l1(img_B, cycle_B)


            G_loss = (
                loss_G_A
                + loss_G_B
                + cycle_A_loss * config.LAMBDA_CYCLE
                + cycle_B_loss * config.LAMBDA_CYCLE
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_B * 0.5 + 0.5, f"saved_images/fake_B_{epoch}_{idx}.png")
            save_image(fake_A * 0.5 + 0.5, f"saved_images/fake_A_{epoch}_{idx}.png")
            save_image(img_B * 0.5 + 0.5, f"saved_images/real_B_{epoch}_{idx}.png")
            save_image(img_A * 0.5 + 0.5, f"saved_images/real_A_{epoch}_{idx}.png")

        loop.set_postfix(B_real=B_reals / (idx + 1), B_fake=B_fakes / (idx + 1))

    writer.add_scalar("Loss/D_B_real", B_reals / len(loader), epoch)
    writer.add_scalar("Loss/D_B_fake", B_fakes / len(loader), epoch)
    writer.add_scalar("Loss/G_loss", G_loss, epoch)

def main():
    os.makedirs("saved_images", exist_ok=True)

    disc_B = Discriminator(in_channels=3).to(config.DEVICE)
    disc_A = Discriminator(in_channels=3).to(config.DEVICE)
    gen_A = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_B = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(
        list(gen_A.parameters()) + list(gen_B.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_gen_B,
            gen_B,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_gen_A,
            gen_A,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_disc_A,
            disc_A,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_disc_B,
            disc_B,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = ImageDataset(
        root_A=os.path.join(config.TRAIN_DIR, "trainA"),
        root_B=os.path.join(config.TRAIN_DIR, "trainB"),
        transform=config.transforms,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    
    val_dataset = ImageDataset(
        root_A=os.path.join(config.TRAIN_DIR, "testA"),
        root_B=os.path.join(config.TRAIN_DIR, "testB"),
        transform=config.transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    g_scaler = torch.amp.GradScaler()
    d_scaler = torch.amp.GradScaler()

    writer = SummaryWriter(config.TENSORBOARD_LOG_DIR)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_A,
            disc_B,
            gen_A,
            gen_B,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
            writer,
            epoch,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_B, opt_gen, filename=config.CHECKPOINT_gen_B)
            save_checkpoint(gen_A, opt_gen, filename=config.CHECKPOINT_gen_A)
            save_checkpoint(disc_A, opt_disc, filename=config.CHECKPOINT_disc_A)
            save_checkpoint(disc_B, opt_disc, filename=config.CHECKPOINT_disc_B)

        save_some_examples(gen_B, gen_A, val_loader, epoch, "saved_images", writer)

if __name__ == "__main__":
    main()