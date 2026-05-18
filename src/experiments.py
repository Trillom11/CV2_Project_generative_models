import torch
import optuna
from src.data import get_dataloaders
from src.models import GeneratorUNet, Discriminator, SpectralNormDiscriminator, AttentionUNet
from src.utils import calculate_metrics

def objective(trial, root_dir, device):
    # Hyperparameter Search Space
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    lambda_pixel = trial.suggest_int("lambda_pixel", 50, 200, step=10)
    batch_size = trial.suggest_categorical("batch_size", [8, 16])
    
    # We fix the architecture to use improvements to search their optimal hyperparams
    use_augmentation = trial.suggest_categorical("use_augmentation", [True, False])
    
    # Init Data
    dataloaders, _ = get_dataloaders(root_dir=root_dir, batch_size=batch_size, use_augmentation=use_augmentation)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # Init Models from scratch
    generator = AttentionUNet().to(device)
    discriminator = SpectralNormDiscriminator().to(device)

    # Optimizers
    criterion_GAN = torch.nn.BCEWithLogitsLoss()
    criterion_pixelwise = torch.nn.L1Loss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Short ultra-loop (e.g. 2 epochs to evaluate convergence trend)
    epochs = 2
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        
        for i, (real_imgs, label_maps) in enumerate(train_loader):
            real_imgs, label_maps = real_imgs.to(device), label_maps.to(device)
            
            patch_size = discriminator(real_imgs, label_maps).shape[2:]
            valid = torch.ones((real_imgs.size(0), 1, *patch_size), device=device)
            fake = torch.zeros((real_imgs.size(0), 1, *patch_size), device=device)

            # Train G
            optimizer_G.zero_grad()
            fake_imgs = generator(label_maps)
            loss_G = criterion_GAN(discriminator(fake_imgs, label_maps), valid) + lambda_pixel * criterion_pixelwise(fake_imgs, real_imgs)
            loss_G.backward()
            optimizer_G.step()

            # Train D
            optimizer_D.zero_grad()
            loss_D_real = criterion_GAN(discriminator(real_imgs, label_maps), valid)
            loss_D_fake = criterion_GAN(discriminator(fake_imgs.detach(), label_maps), fake)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimizer_D.step()

    # Validate to get the final MAE scalar
    generator.eval()
    val_mae_total = 0.0
    with torch.no_grad():
        for real_val, label_val in val_loader:
            real_val, label_val = real_val.to(device), label_val.to(device)
            fake_val = generator(label_val)
            _, _, m = calculate_metrics(real_val, fake_val)
            val_mae_total += m
            
    val_mae = val_mae_total / (len(val_loader) if len(val_loader) > 0 else 1)
    return val_mae

import os
import pickle

def run_optuna_search(root_dir='./MoNuSeg', n_trials=10, device=None, save_path=None):
    if save_path is not None and os.path.exists(save_path):
        print(f"Loading Optuna results from {save_path}...")
        with open(save_path, "rb") as f:
            best_params, optuna_df = pickle.load(f)
        return best_params, optuna_df

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, root_dir, device), n_trials=n_trials)
    
    print("\\n[Optuna Search Completed]")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Validation MAE: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    best_params = study.best_params
    optuna_df = study.trials_dataframe()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump((best_params, optuna_df), f)
        print(f"Optuna results saved to {save_path}")

    return best_params, optuna_df
