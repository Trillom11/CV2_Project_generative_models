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

def train_model(generator, discriminator, dataloaders, epochs=10, device=None, lr=0.0002, lambda_pixel=100, save_path=None, early_stopping_patience=None, use_scheduler=True):
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

    # ReduceLROnPlateau: halves lr after 4 epochs without MAE improvement.
    # Only active for the improved model (use_scheduler=True); baseline keeps fixed lr.
    if use_scheduler:
        scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_G, mode='min', factor=0.5, patience=4, min_lr=1e-6
        )
        scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_D, mode='min', factor=0.5, patience=4, min_lr=1e-6
        )

    histories = {
        'g_loss': [], 'd_loss': [], 'val_g_loss': [], 'val_d_loss': [],
        'val_psnr': [], 'val_ssim': [], 'val_mae': []
    }

    # Early stopping state — monitored metric: val_mae (lower = better)
    best_val_mae      = float('inf')
    early_stop_counter = 0
    best_G_state = None
    best_D_state = None

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
              f"[PSNR: {histories['val_psnr'][-1]:.2f} SSIM: {histories['val_ssim'][-1]:.4f} MAE: {histories['val_mae'][-1]:.4f}]")

        # --- LR scheduler step (on val_mae) ---
        if use_scheduler:
            prev_lr_G = optimizer_G.param_groups[0]['lr']
            scheduler_G.step(histories['val_mae'][-1])
            scheduler_D.step(histories['val_mae'][-1])
            new_lr_G = optimizer_G.param_groups[0]['lr']
            if new_lr_G < prev_lr_G:
                print(f"  [Scheduler] LR reduced: {prev_lr_G:.2e} → {new_lr_G:.2e}")

        # --- Early stopping check (monitors val_mae — lower is better) ---
        if early_stopping_patience is not None:
            current_mae = histories['val_mae'][-1]
            if current_mae < best_val_mae:
                best_val_mae = current_mae
                early_stop_counter = 0
                import copy
                best_G_state = copy.deepcopy(generator.state_dict())
                best_D_state = copy.deepcopy(discriminator.state_dict())
                print(f"  [Early Stopping] ★ New best MAE: {best_val_mae:.4f} — weights saved.")
            else:
                early_stop_counter += 1
                print(f"  [Early Stopping] No MAE improvement for {early_stop_counter}/{early_stopping_patience} epoch(s). Best: {best_val_mae:.4f}")
                if early_stop_counter >= early_stopping_patience:
                    print(f"  [Early Stopping] Triggered at epoch {epoch+1}. Restoring best weights (MAE={best_val_mae:.4f}).")
                    generator.load_state_dict(best_G_state)
                    discriminator.load_state_dict(best_D_state)
                    break

    # Restore best weights even if we finish all epochs without early stopping
    if early_stopping_patience is not None and best_G_state is not None:
        generator.load_state_dict(best_G_state)
        discriminator.load_state_dict(best_D_state)

    if save_path is not None:
        # Save checkpoints
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(generator.state_dict(), f"{save_path}_G.pth")
        torch.save(discriminator.state_dict(), f"{save_path}_D.pth")
        torch.save(histories, f"{save_path}_history.pt")
        print(f"Models and history saved to {save_path}*")

    return histories
