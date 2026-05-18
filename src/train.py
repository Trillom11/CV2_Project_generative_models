# =============================================================================
# Project : Pix2Pix Image-to-Image Translation — MoNuSeg Dataset
# Course  : Computer Vision II · Master in Artificial Intelligence
# School  : Universidade de Santiago de Compostela (USC)
# Authors : Javier Crego Fraguela · Raúl Trillo Martínez
# Year    : 2025–2026
# File    : src/train.py — Adversarial training loop with checkpointing and validation metrics.
#
# Disclaimer
# ----------
# This project was developed through a combination of individual authorship and
# Generative AI assistance (GitHub Copilot / Gemini / Claude). All AI-generated
# code and content was thoroughly reviewed, understood, and validated by the
# authors, who take full academic responsibility for every theoretical choice
# and implementation decision present in this work.
# =============================================================================

import torch
from src.utils import calculate_metrics

import os

def train_model(generator, discriminator, dataloaders, epochs=10, device=None, lr=0.0002, lambda_pixel=100, save_path=None):
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Checkpoint logic
    if save_path is not None and os.path.exists(f"{save_path}_G.pth"):
        print(f"Loading cached model from {save_path}...")
        generator.load_state_dict(torch.load(f"{save_path}_G.pth", map_location=device))
        discriminator.load_state_dict(torch.load(f"{save_path}_D.pth", map_location=device))
        generator = generator.to(device)
        discriminator = discriminator.to(device)
        histories = torch.load(f"{save_path}_history.pt", map_location="cpu", weights_only=False)
        return histories

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    criterion_GAN = torch.nn.BCEWithLogitsLoss()
    criterion_pixelwise = torch.nn.L1Loss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    histories = {
        'g_loss': [], 'd_loss': [], 'val_g_loss': [], 'val_d_loss': [],
        'val_psnr': [], 'val_ssim': [], 'val_mae': []
    }

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        running_g_loss = 0.0
        running_d_loss = 0.0
        
        for i, (real_imgs, label_maps) in enumerate(train_loader):
            real_imgs, label_maps = real_imgs.to(device), label_maps.to(device)
            patch_size = discriminator(real_imgs, label_maps).shape[2:]
            valid = torch.ones((real_imgs.size(0), 1, *patch_size), device=device)
            fake = torch.zeros((real_imgs.size(0), 1, *patch_size), device=device)

            optimizer_G.zero_grad()
            fake_imgs = generator(label_maps)
            pred_fake = discriminator(fake_imgs, label_maps)
            loss_G = criterion_GAN(pred_fake, valid) + lambda_pixel * criterion_pixelwise(fake_imgs, real_imgs)
            loss_G.backward()
            optimizer_G.step()
            running_g_loss += loss_G.item()

            optimizer_D.zero_grad()
            loss_D_real = criterion_GAN(discriminator(real_imgs, label_maps), valid)
            loss_D_fake = criterion_GAN(discriminator(fake_imgs.detach(), label_maps), fake)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimizer_D.step()
            running_d_loss += loss_D.item()

        epoch_g_loss = running_g_loss / len(train_loader)
        epoch_d_loss = running_d_loss / len(train_loader)
        
        generator.eval()
        discriminator.eval()
        val_g_loss, val_d_loss, val_psnr, val_ssim, val_mae = 0.0, 0.0, 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for real_val, label_val in val_loader:
                real_val, label_val = real_val.to(device), label_val.to(device)
                patch_size = discriminator(real_val, label_val).shape[2:]
                valid = torch.ones((real_val.size(0), 1, *patch_size), device=device)
                fake = torch.zeros((real_val.size(0), 1, *patch_size), device=device)
                
                fake_val = generator(label_val)
                val_g_loss += (criterion_GAN(discriminator(fake_val, label_val), valid) + lambda_pixel * criterion_pixelwise(fake_val, real_val)).item()
                val_d_loss += (0.5 * (criterion_GAN(discriminator(real_val, label_val), valid) + criterion_GAN(discriminator(fake_val.detach(), label_val), fake))).item()
                
                p, s, m = calculate_metrics(real_val, fake_val)
                val_psnr += p
                val_ssim += s
                val_mae += m
        
        n_val = len(val_loader) if len(val_loader) > 0 else 1
        
        histories['g_loss'].append(epoch_g_loss)
        histories['d_loss'].append(epoch_d_loss)
        histories['val_g_loss'].append(val_g_loss / n_val)
        histories['val_d_loss'].append(val_d_loss / n_val)
        histories['val_psnr'].append(val_psnr / n_val)
        histories['val_ssim'].append(val_ssim / n_val)
        histories['val_mae'].append(val_mae / n_val)

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"[Train D: {epoch_d_loss:.4f} G: {epoch_g_loss:.4f}] "
              f"[Val D: {histories['val_d_loss'][-1]:.4f} G: {histories['val_g_loss'][-1]:.4f}] "
              f"[PSNR: {histories['val_psnr'][-1]:.2f} SSIM: {histories['val_ssim'][-1]:.4f}]")

    if save_path is not None:
        # Save checkpoints
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(generator.state_dict(), f"{save_path}_G.pth")
        torch.save(discriminator.state_dict(), f"{save_path}_D.pth")
        torch.save(histories, f"{save_path}_history.pt")
        print(f"Models and history saved to {save_path}*")

    return histories
