import matplotlib.pyplot as plt
import numpy as np

def plot_model_comparison(baseline_hist, best_model_hist):
    # Comparar métricas de la última época validada
    labels = ['PSNR', 'SSIM', 'MAE']
    
    b_psnr = baseline_hist['val_psnr'][-1]
    b_ssim = baseline_hist['val_ssim'][-1]
    b_mae = baseline_hist['val_mae'][-1]
    
    m_psnr = best_model_hist['val_psnr'][-1]
    m_ssim = best_model_hist['val_ssim'][-1]
    m_mae = best_model_hist['val_mae'][-1]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Config para PSNR (Higher is better)
    axes[0].bar(['Baseline', 'Mejor Modelo'], [b_psnr, m_psnr], color=['blue', 'green'])
    axes[0].set_title('PSNR (Mayor es mejor)')
    axes[0].set_ylabel('dB')
    
    # Config para SSIM (Higher is better)
    axes[1].bar(['Baseline', 'Mejor Modelo'], [b_ssim, m_ssim], color=['blue', 'green'])
    axes[1].set_title('SSIM (Mayor es mejor)')
    
    # Config para MAE (Lower is better)
    axes[2].bar(['Baseline', 'Mejor Modelo'], [b_mae, m_mae], color=['blue', 'red'])
    axes[2].set_title('MAE (Menor es mejor)')
    
    plt.tight_layout()
    plt.show()
