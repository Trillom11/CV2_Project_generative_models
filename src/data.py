# =============================================================================
# Project : Pix2Pix Image-to-Image Translation — MoNuSeg Dataset
# Course  : Computer Vision II · Master in Artificial Intelligence
# School  : Universidade de Santiago de Compostela (USC)
# Authors : Javier Crego Fraguela · Raúl Trillo Martínez
# Year    : 2025–2026
# File    : src/data.py — Dataset class (MoNuSegDataset), transforms, and DataLoader factory for train/val/test splits.
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
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

class MoNuSegDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.filenames = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg'))]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.filenames[idx])
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        real_img = img.crop((0, 0, w // 2, h))
        label_map = img.crop((w // 2, 0, w, h))
        
        # We need to apply identical spatial transforms to both images to keep them aligned
        # In a real scenario for random flip/rotation, we concat them, transform, then split.
        # For simplicity in this assignment, we handle them as pairs if complex aug is needed.
        if self.transform:
            # Note: For random augmentation (like flip), to ensure the real and label are flipped exactly the same, 
            # we concatenate them, transform, then split back. 
            combined = Image.new('RGB', (real_img.width + label_map.width, real_img.height))
            combined.paste(real_img, (0, 0))
            combined.paste(label_map, (real_img.width, 0))
            
            combined = self.transform(combined)
            
            # Split back (assuming transform outputs a CxHxW tensor where W is double)
            c, h_t, w_t = combined.shape
            real_img = combined[:, :, :w_t//2]
            label_map = combined[:, :, w_t//2:]
            
        return real_img, label_map

def get_transforms():
    # Baseline transforms
    return transforms.Compose([
        transforms.Resize((256, 512)), # 256x256 per image, concatenated it's 256x512
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def get_augmented_transforms():
    # Augmented transforms
    return transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def get_dataloaders(root_dir='./MoNuSeg', batch_size=16, use_augmentation=False):
    dataset_transforms = get_augmented_transforms() if use_augmentation else get_transforms()
    full_dataset = MoNuSegDataset(root_dir=root_dir, transform=dataset_transforms)
    
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return {"train": train_loader, "val": val_loader, "test": test_loader}, full_dataset
