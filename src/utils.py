# =============================================================================
# Project : Pix2Pix Image-to-Image Translation — MoNuSeg Dataset
# Course  : Computer Vision II · Master in Artificial Intelligence
# School  : Universidade de Santiago de Compostela (USC)
# Authors : Javier Crego Fraguela · Raúl Trillo Martínez
# Year    : 2025–2026
# File    : src/utils.py — Metric computation (PSNR, SSIM, MAE) and matplotlib visualisation helpers.
#
# Disclaimer
# ----------
# This project was developed through a combination of individual authorship and
# Generative AI assistance (GitHub Copilot / Gemini / Claude). All AI-generated
# code and content was thoroughly reviewed, understood, and validated by the
# authors, who take full academic responsibility for every theoretical choice
# and implementation decision present in this work.
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(real_imgs, fake_imgs):
    real_np = ((real_imgs * 0.5) + 0.5).detach().cpu().permute(0, 2, 3, 1).numpy()
    fake_np = ((fake_imgs * 0.5) + 0.5).detach().cpu().permute(0, 2, 3, 1).numpy()
    
    batch_psnr = []
    batch_ssim = []
    batch_mae  = []
    
    for i in range(real_np.shape[0]):
        r = real_np[i]
        f = fake_np[i]
        batch_mae.append(np.mean(np.abs(r - f)))
        batch_psnr.append(psnr(r, f, data_range=1.0))
        batch_ssim.append(ssim(r, f, data_range=1.0, channel_axis=-1))
        
    return np.mean(batch_psnr), np.mean(batch_ssim), np.mean(batch_mae)

def visualize_batch(dataloader, num_images=4):
    real_imgs, label_maps = next(iter(dataloader))
    real_imgs = (real_imgs[:num_images] * 0.5) + 0.5
    label_maps = (label_maps[:num_images] * 0.5) + 0.5
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
    for i in range(num_images):
        axes[0, i].imshow(real_imgs[i].permute(1, 2, 0).cpu().numpy())
        axes[0, i].set_title("Real Image")
        axes[0, i].axis("off")
        axes[1, i].imshow(label_maps[i].permute(1, 2, 0).cpu().numpy())
        axes[1, i].set_title("Label Map")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.show()

def plot_metrics(histories):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(histories.get('d_loss', []), label="Train D L")
    axes[0].plot(histories.get('val_d_loss', []), label="Val D L", linestyle='--')
    axes[0].plot(histories.get('g_loss', []), label="Train G L")
    axes[0].plot(histories.get('val_g_loss', []), label="Val G L", linestyle='--')
    axes[0].set_title("Discriminator & Generator Losses")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(histories.get('val_psnr', []), label="Val PSNR", color='g')
    axes[1].set_title("Peak Signal-to-Noise Ratio (PSNR)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[2].plot(histories.get('val_ssim', []), label="Val SSIM", color='orange')
    axes[2].plot(histories.get('val_mae', []), label="Val MAE (L1)", color='r')
    axes[2].set_title("Structural Similarity & Absolute Error")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    plt.tight_layout()
    plt.show()

def visualize_progress(label_maps, fake_imgs, real_imgs, num_images=3):
    labels = (label_maps[:num_images] * 0.5) + 0.5
    fakes = (fake_imgs[:num_images] * 0.5) + 0.5
    reals = (real_imgs[:num_images] * 0.5) + 0.5
    fig, axes = plt.subplots(num_images, 3, figsize=(10, 4*num_images))
    for i in range(num_images):
        axes[i, 0].imshow(labels[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 0].set_title("Input (Label Map)")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(fakes[i].detach().permute(1, 2, 0).cpu().numpy())
        axes[i, 1].set_title("Generated Image")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(reals[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 2].set_title("Ground Truth (Real)")
        axes[i, 2].axis("off")
    plt.tight_layout()
    plt.show()
