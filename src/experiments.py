# =============================================================================
# Project : Pix2Pix Image-to-Image Translation — MoNuSeg Dataset
# Course  : Computer Vision II · Master in Artificial Intelligence
# School  : Universidade de Santiago de Compostela (USC)
# Authors : Javier Crego Fraguela · Raúl Trillo Martínez
# Year    : 2025–2026
# File    : src/experiments.py — Bayesian hyperparameter search using Optuna (TPE sampler).
#           Full study persistence via SQLite; best-trial generator weights saved to disk.
#
# Disclaimer
# ----------
# This project was developed through a combination of individual authorship and
# Generative AI assistance (GitHub Copilot / Gemini / Claude). All AI-generated
# code and content was thoroughly reviewed, understood, and validated by the
# authors, who take full academic responsibility for every theoretical choice
# and implementation decision present in this work.
# =============================================================================

import os
import pickle
import copy

import torch
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)  # keep output clean

from src.data import get_dataloaders
from src.models import AttentionUNet, SpectralNormDiscriminator
from src.utils import calculate_metrics


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial, root_dir, device, best_weights_path=None):
    """Single Optuna trial: 4-epoch training with val-MAE as objective."""

    # --- Hyperparameter search space ---
    lr            = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    lambda_pixel  = trial.suggest_int("lambda_pixel", 50, 200, step=10)
    batch_size    = trial.suggest_categorical("batch_size", [8, 16])
    dropout_rate  = trial.suggest_float("dropout_rate", 0.3, 0.6, step=0.1)
    use_augmentation = trial.suggest_categorical("use_augmentation", [True, False])

    # --- Data ---
    dataloaders, _ = get_dataloaders(
        root_dir=root_dir, batch_size=batch_size, use_augmentation=use_augmentation
    )
    train_loader = dataloaders["train"]
    val_loader   = dataloaders["val"]

    # --- Models ---
    generator     = AttentionUNet(dropout_rate=dropout_rate).to(device)
    discriminator = SpectralNormDiscriminator().to(device)

    # --- Optimizers & losses ---
    criterion_GAN      = torch.nn.BCEWithLogitsLoss()
    criterion_pixelwise = torch.nn.L1Loss()
    optimizer_G = torch.optim.Adam(generator.parameters(),     lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # --- 4-epoch training loop ---
    epochs = 4
    t_num  = trial.number
    print(f"  ┌─ Trial #{t_num}  lr={lr:.2e}  λ={lambda_pixel}  bs={batch_size}  dr={dropout_rate:.1f}  aug={use_augmentation}")
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        running_g, running_d = 0.0, 0.0
        n_batches = 0

        for real_imgs, label_maps in train_loader:
            real_imgs, label_maps = real_imgs.to(device), label_maps.to(device)

            patch_size = discriminator(real_imgs, label_maps).shape[2:]
            valid = torch.ones((real_imgs.size(0), 1, *patch_size), device=device)
            fake  = torch.zeros((real_imgs.size(0), 1, *patch_size), device=device)

            # Generator step
            optimizer_G.zero_grad()
            fake_imgs = generator(label_maps)
            loss_G = (criterion_GAN(discriminator(fake_imgs, label_maps), valid)
                      + lambda_pixel * criterion_pixelwise(fake_imgs, real_imgs))
            loss_G.backward()
            optimizer_G.step()
            running_g += loss_G.item()

            # Discriminator step
            optimizer_D.zero_grad()
            loss_D = 0.5 * (
                criterion_GAN(discriminator(real_imgs, label_maps), valid)
                + criterion_GAN(discriminator(fake_imgs.detach(), label_maps), fake)
            )
            loss_D.backward()
            optimizer_D.step()
            running_d += loss_D.item()
            n_batches += 1

        print(f"  │  Epoch [{epoch+1}/{epochs}]  G: {running_g/max(n_batches,1):.4f}  D: {running_d/max(n_batches,1):.4f}")

    # --- Validation ---
    generator.eval()
    val_mae_total = 0.0
    with torch.no_grad():
        for real_val, label_val in val_loader:
            real_val, label_val = real_val.to(device), label_val.to(device)
            fake_val = generator(label_val)
            _, _, m = calculate_metrics(real_val, fake_val)
            val_mae_total += m

    n_val   = len(val_loader) if len(val_loader) > 0 else 1
    val_mae = val_mae_total / n_val

    # --- Trial summary ---
    try:
        prev_best = trial.study.best_value
        is_new_best = val_mae <= prev_best
    except Exception:
        prev_best = float('inf')
        is_new_best = True

    star = " ★ NEW BEST" if is_new_best else f"  (best so far: {prev_best:.4f})"
    print(f"  └─ Val MAE: {val_mae:.4f}{star}")
    print()

    # --- Persist best generator weights ---
    # After this trial's value is reported, check whether it is the new best.
    # We use the study's current best_value (which has already been updated
    # by Optuna before the callback fires, but since we're inside the objective
    # we compare manually).
    if best_weights_path is not None:
        is_first_trial = (trial.number == 0)
        try:
            current_best = trial.study.best_value  # best *so far* (may not include this trial yet)
            is_new_best  = val_mae <= current_best
        except Exception:
            is_new_best = True  # fallback for first trial edge-case

        if is_first_trial or is_new_best:
            os.makedirs(os.path.dirname(best_weights_path), exist_ok=True)
            torch.save(copy.deepcopy(generator.state_dict()), best_weights_path)

    return val_mae


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_optuna_search(
    root_dir='./MoNuSeg',
    n_trials=10,
    device=None,
    save_path=None,           # path for lightweight (best_params, optuna_df) pickle
    db_path=None,             # path for SQLite study storage, e.g. "checkpoints/optuna_study.db"
    best_weights_path=None,   # path to save best-trial generator weights, e.g. "checkpoints/optuna_best_trial_G.pth"
):
    """
    Run (or reload) a Bayesian hyperparameter search with Optuna.

    Persistence strategy
    --------------------
    * Full study → SQLite (``db_path``) — all trials, params, timing.  Survives
      kernel restarts; can be loaded and visualised offline.
    * Lightweight backup → pickle (``save_path``) — (best_params, optuna_df).
    * Best generator weights → ``best_weights_path``.

    If the SQLite DB already exists the study is loaded from disk (no re-training).
    If only the pickle exists (legacy) it is loaded as a fallback.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    study_name = "pix2pix_optuna"

    # ------------------------------------------------------------------
    # 1. Try to load from SQLite (primary persistence)
    # ------------------------------------------------------------------
    if db_path is not None and os.path.exists(db_path):
        print(f"[Optuna] Loading existing study from SQLite: {db_path}\n")
        storage = f"sqlite:///{db_path}"
        study   = optuna.load_study(study_name=study_name, storage=storage)

        # Replay formatted output for every completed trial
        running_best = float('inf')
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        completed.sort(key=lambda t: t.number)

        for t in completed:
            p = t.params
            is_best = t.value <= running_best
            if is_best:
                running_best = t.value

            # Header line with hyperparameters
            print(f"  ┌─ Trial #{t.number}"
                  f"  lr={p.get('lr', '?'):.2e}"
                  f"  λ={p.get('lambda_pixel', '?')}"
                  f"  bs={p.get('batch_size', '?')}"
                  f"  dr={p.get('dropout_rate', '?')}"
                  f"  aug={p.get('use_augmentation', '?')}")

            # Duration info instead of per-epoch losses (not stored in SQLite)
            duration = t.duration.total_seconds() if t.duration else 0
            print(f"  │  (loaded from cache — {duration:.0f}s training time)")

            # Result line
            star = " ★ NEW BEST" if is_best else f"  (best so far: {running_best:.4f})"
            print(f"  └─ Val MAE: {t.value:.4f}{star}")

            # Separator
            print(f"{'─'*56}")
            print(f" Trial #{t.number} done  ({t.number+1}/{len(completed)})"
                  f"  MAE={t.value:.4f}"
                  f"  Best=#{study.best_trial.number} ({study.best_value:.4f})")
            print(f"{'─'*56}\n")

        best_params = study.best_params
        optuna_df   = study.trials_dataframe()
        return best_params, optuna_df

    # ------------------------------------------------------------------
    # 2. Fallback: legacy pickle (backward compatibility)
    # ------------------------------------------------------------------
    if save_path is not None and os.path.exists(save_path) and db_path is None:
        print(f"[Optuna] Loading cached results from pickle: {save_path}")
        with open(save_path, "rb") as f:
            best_params, optuna_df = pickle.load(f)
        return best_params, optuna_df

    # ------------------------------------------------------------------
    # 3. Run a fresh search
    # ------------------------------------------------------------------
    storage = None
    if db_path is not None:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=storage,
        load_if_exists=True,   # safe to resume interrupted runs
    )

    def _trial_callback(study, trial):
        n_done  = len([t for t in study.trials if t.state.is_finished()])
        n_total = n_trials
        print(f"{'─'*56}")
        print(f" Trial #{trial.number} done  ({n_done}/{n_total})  "
              f"MAE={trial.value:.4f}  "
              f"Best=#{study.best_trial.number} ({study.best_value:.4f})")
        print(f"{'─'*56}")
        print()

    study.optimize(
        lambda trial: objective(trial, root_dir, device, best_weights_path=best_weights_path),
        n_trials=n_trials,
        callbacks=[_trial_callback],
    )

    # ------------------------------------------------------------------
    # 4. Report results
    # ------------------------------------------------------------------
    print("\n[Optuna Search Completed]")
    print(f"  Best Trial  : #{study.best_trial.number}")
    print(f"  Best Val MAE: {study.best_value:.4f}")
    print("  Best Params :")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    best_params = study.best_params
    optuna_df   = study.trials_dataframe()

    # ------------------------------------------------------------------
    # 5. Persist lightweight pickle backup
    # ------------------------------------------------------------------
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump((best_params, optuna_df), f)
        print(f"[Optuna] Pickle backup saved → {save_path}")

    if db_path is not None:
        print(f"[Optuna] Full study persisted in SQLite → {db_path}")
    if best_weights_path is not None and os.path.exists(best_weights_path):
        print(f"[Optuna] Best-trial generator weights saved → {best_weights_path}")

    return best_params, optuna_df
