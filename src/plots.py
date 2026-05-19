# =============================================================================
# Project : Pix2Pix Image-to-Image Translation — MoNuSeg Dataset
# Course  : Computer Vision II · Master in Artificial Intelligence
# School  : Universidade de Santiago de Compostela (USC)
# Authors : Javier Crego Fraguela · Raúl Trillo Martínez
# Year    : 2025–2026
# File    : src/plots.py — Model comparison bar charts (Baseline vs. Best Model).
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
import matplotlib.gridspec as gridspec
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


def plot_optuna_analysis(optuna_df, best_params):
    """
    Comparative analysis of all Optuna trials.

    Four panels:
      1. Optimization history  — val MAE per trial + running best.
      2. Learning rate vs MAE  — scatter coloured by dropout_rate.
      3. Lambda_pixel vs MAE   — scatter coloured by batch_size.
      4. Dropout rate vs MAE   — scatter + best-trial annotation.
    """
    # ---- Robust column resolution ------------------------------------------
    # trials_dataframe() column names differ across Optuna versions:
    #   older : 'params_lr', 'params_lambda_pixel', …, 'value'
    #   newer : 'params_lr', …,                        'values_0' or 'value'
    col_map = {}
    for col in optuna_df.columns:
        if 'value' in col.lower() and 'param' not in col.lower():
            col_map.setdefault('value', col)   # first match wins
        for hp in ('lr', 'lambda_pixel', 'batch_size', 'dropout_rate', 'use_augmentation'):
            if col == f'params_{hp}':
                col_map[hp] = col

    df = optuna_df[optuna_df[col_map['value']].notna()].copy()
    values   = df[col_map['value']].values
    trial_no = df['number'].values
    best_idx = int(np.argmin(values))

    # Colour palettes
    BEST_COLOUR  = '#f97316'   # orange star for best trial
    HIST_COLOUR  = '#6366f1'   # indigo bars
    BEST_LINE    = '#f97316'

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Optuna Hyperparameter Search — Trial Analysis', fontsize=16, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ------------------------------------------------------------------
    # Panel 1 — Optimization History
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    running_best = np.minimum.accumulate(values)

    ax1.bar(trial_no, values, color=HIST_COLOUR, alpha=0.6, label='Trial MAE')
    ax1.plot(trial_no, running_best, color=BEST_LINE, linewidth=2,
             marker='o', markersize=4, label='Running best')
    ax1.scatter(trial_no[best_idx], values[best_idx], marker='*',
                s=260, color=BEST_COLOUR, zorder=5, label=f'Best (trial #{trial_no[best_idx]})')
    ax1.set_xlabel('Trial #')
    ax1.set_ylabel('Validation MAE')
    ax1.set_title('Optimization History')
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # ------------------------------------------------------------------
    # Panel 2 — Learning Rate vs MAE
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    if 'lr' in col_map and 'dropout_rate' in col_map:
        lr_vals      = df[col_map['lr']].values
        dropout_vals = df[col_map['dropout_rate']].values
        sc = ax2.scatter(lr_vals, values, c=dropout_vals,
                         cmap='plasma', s=90, alpha=0.85, zorder=3)
        ax2.scatter(lr_vals[best_idx], values[best_idx], marker='*',
                    s=300, color=BEST_COLOUR, zorder=5, label='Best trial')
        ax2.set_xscale('log')
        ax2.set_xlabel('Learning Rate (log scale)')
        ax2.set_ylabel('Validation MAE')
        ax2.set_title('Learning Rate vs MAE\n(colour = dropout rate)')
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        fig.colorbar(sc, ax=ax2, label='dropout_rate')
    else:
        ax2.text(0.5, 0.5, 'lr / dropout_rate data not available',
                 ha='center', va='center', transform=ax2.transAxes)

    # ------------------------------------------------------------------
    # Panel 3 — Lambda Pixel vs MAE
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    if 'lambda_pixel' in col_map and 'batch_size' in col_map:
        lp_vals  = df[col_map['lambda_pixel']].values
        bs_vals  = df[col_map['batch_size']].values
        # Map batch_size to two colours
        bs_unique = sorted(set(bs_vals))
        palette   = ['#818cf8', '#f97316']
        bs_colour = [palette[bs_unique.index(b) % len(palette)] for b in bs_vals]

        for b, p in zip(bs_unique, palette):
            mask = bs_vals == b
            ax3.scatter(lp_vals[mask], values[mask], color=p, s=90,
                        alpha=0.85, label=f'batch={int(b)}', zorder=3)
        ax3.scatter(lp_vals[best_idx], values[best_idx], marker='*',
                    s=300, color=BEST_COLOUR, zorder=5, label='Best trial')
        ax3.set_xlabel('Lambda Pixel')
        ax3.set_ylabel('Validation MAE')
        ax3.set_title('Lambda Pixel vs MAE\n(colour = batch size)')
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'lambda_pixel / batch_size data not available',
                 ha='center', va='center', transform=ax3.transAxes)

    # ------------------------------------------------------------------
    # Panel 4 — Dropout Rate vs MAE
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    if 'dropout_rate' in col_map:
        dr_vals = df[col_map['dropout_rate']].values
        ax4.scatter(dr_vals, values, color=HIST_COLOUR, s=90, alpha=0.85, zorder=3)
        ax4.scatter(dr_vals[best_idx], values[best_idx], marker='*',
                    s=300, color=BEST_COLOUR, zorder=5,
                    label=f'Best  (dr={dr_vals[best_idx]:.1f}, MAE={values[best_idx]:.4f})')
        # Annotate each point with trial number
        for tn, dr, v in zip(trial_no, dr_vals, values):
            ax4.annotate(f'#{tn}', (dr, v), textcoords='offset points',
                         xytext=(6, 4), fontsize=7, alpha=0.7)
        ax4.set_xlabel('Dropout Rate')
        ax4.set_ylabel('Validation MAE')
        ax4.set_title('Dropout Rate vs MAE')
        ax4.legend(fontsize=8)
        ax4.grid(alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'dropout_rate data not available',
                 ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout()
    plt.show()
