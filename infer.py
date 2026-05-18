"""
infer.py — Pix2Pix MoNuSeg Demo Script
=======================================
Usage:
    python3 infer.py <image_path> [--weights <path>] [--model baseline|attention]

Example:
    python3 infer.py samples/sample_01_monuseg.png

The generated image is saved automatically to outputs/ and a side-by-side
matplotlib figure (Label Map | Generated | Ground Truth) is displayed.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image

# Import the Generator architectures from the local src package
from src.models import AttentionUNet, GeneratorUNet

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS    = "checkpoints/best_model_G.pth"
DEFAULT_FALLBACK   = "checkpoints/baseline_G.pth"
OUTPUT_DIR         = "outputs"

# ── Image helpers ─────────────────────────────────────────────────────────────

def load_paired_image(img_path):
    """
    Load a (possibly paired) image and return:
      - label_pil : the label map (right half if paired, otherwise the full image)
      - real_pil  : the ground-truth tissue image (left half if paired, else None)
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if w >= h * 1.9:
        real_pil  = img.crop((0, 0, w // 2, h))
        label_pil = img.crop((w // 2, 0, w, h))
    else:
        real_pil  = None
        label_pil = img
    return label_pil, real_pil


def to_tensor(pil_img):
    """Resize to 256×256, convert to tensor and normalize to [-1, 1]."""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform(pil_img).unsqueeze(0)   # add batch dimension


def tensor_to_pil(tensor):
    """Denormalize a [1, C, H, W] tensor from [-1, 1] to a PIL Image."""
    tensor = (tensor.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(tensor.cpu())

# ── Visualisation ─────────────────────────────────────────────────────────────

def show_results(label_pil, generated_pil, real_pil, output_path):
    """Display a side-by-side comparison and mark the saved output path."""
    n_cols = 3 if real_pil is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    fig.suptitle("Pix2Pix — MoNuSeg Inference Demo", fontsize=14, fontweight="bold")

    axes[0].imshow(label_pil)
    axes[0].set_title("Input — Label Map", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(generated_pil)
    axes[1].set_title("Generated Image", fontweight="bold")
    axes[1].axis("off")

    if real_pil is not None:
        axes[2].imshow(real_pil)
        axes[2].set_title("Ground Truth", fontweight="bold")
        axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path.replace(".png", "_comparison.png").replace(".jpg", "_comparison.png"), dpi=150)
    plt.show()
    print(f"[✓] Comparison figure saved alongside the output.")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pix2Pix inference demo — generates a tissue image from a label map.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input image (label map or paired MoNuSeg image).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS,
        help=f"Path to the generator .pth weights file.\n(default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="attention",
        choices=["baseline", "attention"],
        help="Generator architecture to use (default: attention).",
    )
    args = parser.parse_args()

    # ── Validate input ────────────────────────────────────────────────────────
    if not os.path.exists(args.input):
        sys.exit(f"[Error] Input image not found: {args.input}")

    weights_path = args.weights
    if not os.path.exists(weights_path):
        print(f"[Warning] Weights not found at '{weights_path}'. Trying fallback: {DEFAULT_FALLBACK}")
        weights_path = DEFAULT_FALLBACK
        if not os.path.exists(weights_path):
            sys.exit(f"[Error] No weights file found. Please run training first.")

    # ── Prepare output path ───────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename  = os.path.splitext(os.path.basename(args.input))[0]
    out_path  = os.path.join(OUTPUT_DIR, f"{filename}_generated.png")

    # ── Load image ────────────────────────────────────────────────────────────
    print(f"[→] Loading image: {args.input}")
    label_pil, real_pil = load_paired_image(args.input)
    input_tensor = to_tensor(label_pil)

    # ── Load model ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[→] Device: {device} | Architecture: {args.model}")

    model = AttentionUNet().to(device) if args.model == "attention" else GeneratorUNet().to(device)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    except Exception as e:
        sys.exit(f"[Error] Could not load weights: {e}")

    model.eval()

    # ── Inference ─────────────────────────────────────────────────────────────
    print("[→] Running inference...")
    with torch.no_grad():
        generated_tensor = model(input_tensor.to(device))

    # ── Save & display ────────────────────────────────────────────────────────
    generated_pil = tensor_to_pil(generated_tensor)
    generated_pil.save(out_path)
    print(f"[✓] Generated image saved to: {out_path}")

    show_results(label_pil, generated_pil, real_pil, out_path)


if __name__ == "__main__":
    main()
