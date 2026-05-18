# =============================================================================
# Project : Pix2Pix Image-to-Image Translation — MoNuSeg Dataset
# Course  : Computer Vision II · Master in Artificial Intelligence
# School  : Universidade de Santiago de Compostela (USC)
# Authors : Javier Crego Fraguela · Raúl Trillo Martínez
# Year    : 2025–2026
# File    : src/inference.py — Cross-domain evaluation helper used in the notebook demo phase.
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
from src.utils import calculate_metrics, visualize_progress

def run_cross_domain_evaluation(generator, dataloader, device=None, num_images_to_plot=3):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    generator = generator.to(device)
    generator.eval()
    
    val_psnr_total = 0.0
    val_ssim_total = 0.0
    val_mae_total = 0.0
    
    plotted = False
    
    with torch.no_grad():
        for i, (real_imgs, label_maps) in enumerate(dataloader):
            real_imgs, label_maps = real_imgs.to(device), label_maps.to(device)
            fake_imgs = generator(label_maps)
            
            p, s, m = calculate_metrics(real_imgs, fake_imgs)
            val_psnr_total += p
            val_ssim_total += s
            val_mae_total += m
            
            if not plotted:
                print("Visualizing model generalization on unseen domain...")
                visualize_progress(label_maps, fake_imgs, real_imgs, num_images=num_images_to_plot)
                plotted = True
                
    n_val = len(dataloader) if len(dataloader) > 0 else 1
    
    avg_psnr = val_psnr_total / n_val
    avg_ssim = val_ssim_total / n_val
    avg_mae = val_mae_total / n_val
    
    print("\n--- Cross-Domain Evaluation Metrics ---")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average MAE:  {avg_mae:.4f}")
    
    return avg_psnr, avg_ssim, avg_mae
