"""
Complete U-Net training pipeline for Oxford-IIIT Pet segmentation.

Runs three experiments comparing loss functions:
  1. Cross-Entropy only
  2. Dice only
  3. Cross-Entropy + Dice

All use the same hyperparameters; only the loss function differs.
Produces a comparison plot of training curves and final mIoU.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for remote servers
import matplotlib.pyplot as plt

# Local imports
from model import UNet, count_parameters
from loss import DiceLoss, CombinedLoss, compute_miou

# ─── Configuration ───────────────────────────────────────────────────────────

class Config:
    """All hyperparameters live here — easy to modify for experiments."""

    # Paths
    data_root = os.environ.get(
        "PET_DATA",
        "/inspire/hdd/project/fdu-aidake-cfff/public/zhouhaowen/unet-segmentation/data",
    )
    output_dir = os.environ.get(
        "OUTPUT_DIR",
        "/inspire/hdd/project/fdu-aidake-cfff/public/zhouhaowen/unet-segmentation/output",
    )

    # Data
    image_size = 256
    n_channels = 3
    n_classes = 3
    label_offset = -1  # Oxford labels {1,2,3} → {0,1,2}

    # Training
    epochs = 30
    batch_size = 16
    learning_rate = 1e-3
    weight_decay = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loss configs for three experiments
    # Keys are experiment names (used for directories & plot legend)
    loss_configs = {
        "CE_only":   {"type": "ce",   "ce_weight": 1.0, "dice_weight": 0.0},
        "Dice_only": {"type": "dice", "ce_weight": 0.0, "dice_weight": 1.0},
        "CE_Dice":   {"type": "both", "ce_weight": 1.0, "dice_weight": 1.0},
    }

    # Seed
    seed = 42

    # Validation interval (validate every N epochs)
    val_interval = 1


# ─── Dataset Wrapper ─────────────────────────────────────────────────────────

class PetDataset(Dataset):
    """
    Wraps OxfordIIITPet with label mapping {1,2,3}→{0,1,2} and resizing.
    """

    def __init__(self, root, split="trainval", image_size=256):
        self.image_size = image_size
        # Image transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
        ])
        # Download = False because we manually placed the data
        # If it fails, we'll prompt the user
        self.dataset = OxfordIIITPet(
            root=root, split=split, target_types="segmentation", download=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        # mask is a PIL image with values {1, 2, 3}

        img = self.img_transform(img)
        mask = self.mask_transform(mask)

        # Convert PIL to tensor and map labels {1,2,3} → {0,1,2}
        mask = torch.as_tensor(np.array(mask), dtype=torch.long) - 1

        return img, mask


def get_dataloaders(cfg: Config):
    """Create train/val dataloaders."""
    print("=" * 60)
    print("Loading Oxford-IIIT Pet Dataset...")
    print(f"  Data root: {cfg.data_root}")

    # Use trainval split for training and test split for validation
    train_dataset = PetDataset(cfg.data_root, split="trainval", image_size=cfg.image_size)
    val_dataset = PetDataset(cfg.data_root, split="test", image_size=cfg.image_size)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, val_loader


# ─── Loss Factory ────────────────────────────────────────────────────────────

def build_loss(cfg: Config, exp_name: str) -> nn.Module:
    """Build loss function based on experiment configuration."""
    lc = cfg.loss_configs[exp_name]
    ltype = lc["type"]

    if ltype == "ce":
        return nn.CrossEntropyLoss()
    elif ltype == "dice":
        return DiceLoss()
    elif ltype == "both":
        return CombinedLoss(
            ce_weight=lc["ce_weight"],
            dice_weight=lc["dice_weight"],
        )
    else:
        raise ValueError(f"Unknown loss type: {ltype}")


# ─── Training Epoch ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, scaler=None):
    """Run one training epoch, return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, masks)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        if batch_idx % 20 == 0:
            print(f"    Batch [{batch_idx:3d}/{num_batches}]  Loss: {loss.item():.4f}")

    return total_loss / num_batches


# ─── Validation Epoch ────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation, return (avg_loss, mIoU)."""
    model.eval()
    total_loss = 0.0
    all_mious = []
    num_batches = len(loader)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, masks)
        total_loss += loss.item()

        miou = compute_miou(logits, masks, n_classes=3)
        all_mious.append(miou)

    return total_loss / num_batches, np.mean(all_mious)


# ─── Single Experiment ───────────────────────────────────────────────────────

def run_experiment(cfg: Config, exp_name: str) -> dict:
    """
    Run one training experiment with a specific loss configuration.
    Returns training history dict.
    """
    print("\n" + "=" * 60)
    print(f"  Experiment: {exp_name}")
    print("=" * 60)

    # Setup
    torch.manual_seed(cfg.seed + hash(exp_name) % 1000)
    device = torch.device(cfg.device)
    print(f"  Device: {device}")

    # Model
    model = UNet(n_channels=cfg.n_channels, n_classes=cfg.n_classes).to(device)
    total_params = count_parameters(model)
    print(f"  Model parameters: {total_params:,}")

    # Loss & optimizer
    criterion = build_loss(cfg, exp_name)
    scaler = torch.amp.GradScaler("cuda")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    # Data
    train_loader, val_loader = get_dataloaders(cfg)

    # Training loop
    history = {
        "exp_name": exp_name,
        "train_losses": [],
        "val_losses": [],
        "val_mious": [],
        "best_miou": 0.0,
        "best_epoch": -1,
        "params": total_params,
    }

    exp_dir = os.path.join(cfg.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    start_time = time.time()
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()

        print(f"\n  Epoch {epoch:2d}/{cfg.epochs}")
        print(f"  {'─' * 40}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler)

        # Validate every val_interval epochs (and always on the last epoch)
        do_val = (epoch % cfg.val_interval == 0) or (epoch == cfg.epochs)
        if do_val:
            val_loss, val_miou = validate(model, val_loader, criterion, device)
        else:
            val_loss, val_miou = 0.0, 0.0

        epoch_time = time.time() - epoch_start
        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["val_mious"].append(val_miou)

        print(f"  {'─' * 40}")
        if do_val:
            print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")
            print(f"  Val mIoU:   {val_miou:.4f}  |  Time: {epoch_time:.1f}s")
        else:
            print(f"  Train Loss: {train_loss:.4f}  |  Time: {epoch_time:.1f}s")

        # Save best model
        if do_val and val_miou > history["best_miou"]:
            history["best_miou"] = val_miou
            history["best_epoch"] = epoch
            best_path = os.path.join(exp_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "miou": val_miou,
                "loss": val_loss,
                "exp_name": exp_name,
            }, best_path)
            print(f"  ★ New best model saved (mIoU: {val_miou:.4f})")

    total_time = time.time() - start_time
    print(f"\n  Finished {exp_name} in {total_time:.1f}s")
    print(f"  Best mIoU: {history['best_miou']:.4f} @ epoch {history['best_epoch']}")

    # Save history
    history_path = os.path.join(exp_dir, "history.json")
    with open(history_path, "w") as f:
        # Convert numpy floats to Python floats
        serializable = {
            k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in history.items()
        }
        json.dump(history, f, indent=2)

    return history


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_comparison(histories: list, output_dir: str):
    """Plot training curves and final mIoU bar chart for all experiments."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {"CE_only": "#E74C3C", "Dice_only": "#3498DB", "CE_Dice": "#2ECC71"}
    markers = {"CE_only": "o", "Dice_only": "s", "CE_Dice": "^"}

    epochs = range(1, len(histories[0]["train_losses"]) + 1)

    # ── Plot 1: Training Loss ──
    ax = axes[0]
    for h in histories:
        ax.plot(epochs, h["train_losses"],
                label=h["exp_name"], color=colors.get(h["exp_name"], "gray"),
                marker=markers.get(h["exp_name"], "."), markevery=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Validation mIoU ──
    ax = axes[1]
    for h in histories:
        ax.plot(epochs, h["val_mious"],
                label=h["exp_name"], color=colors.get(h["exp_name"], "gray"),
                marker=markers.get(h["exp_name"], "."), markevery=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation mIoU")
    ax.set_title("Validation mIoU Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 3: Best mIoU bar chart ──
    ax = axes[2]
    names = [h["exp_name"] for h in histories]
    best_mious = [h["best_miou"] for h in histories]
    bar_colors = [colors.get(n, "gray") for n in names]
    bars = ax.bar(names, best_mious, color=bar_colors, width=0.5, edgecolor="black")
    ax.set_ylabel("Best Validation mIoU")
    ax.set_title("Best mIoU Comparison")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, best_mious):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to: {plot_path}")
    plt.close()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="U-Net Oxford Pets Segmentation")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--val-interval", type=int, default=None,
                        help="Validate every N epochs (default: 1)")
    parser.add_argument("--experiments", type=str, nargs="+",
                        choices=["CE_only", "Dice_only", "CE_Dice"],
                        default=None, help="Which experiments to run (default: all)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override data root directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    cfg = Config()

    # Override from command line
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.val_interval is not None:
        cfg.val_interval = args.val_interval
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    experiments = args.experiments or list(cfg.loss_configs.keys())

    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 60)
    print("  U-Net Segmentation — Oxford-IIIT Pet Dataset")
    print("=" * 60)
    print(f"  Image size:   {cfg.image_size}×{cfg.image_size}")
    print(f"  Classes:      {cfg.n_classes} (foreground / background / boundary)")
    print(f"  Epochs:       {cfg.epochs}")
    print(f"  Batch size:   {cfg.batch_size}")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  Device:       {cfg.device}")
    print(f"  Output dir:   {cfg.output_dir}")
    print(f"  Experiments:  {', '.join(experiments)}")
    print("=" * 60)

    # Run experiments
    histories = []
    for exp_name in experiments:
        h = run_experiment(cfg, exp_name)
        histories.append(h)
        print(f"\n  ✓ {exp_name} completed (best mIoU: {h['best_miou']:.4f})")

    # Plot comparison
    plot_comparison(histories, cfg.output_dir)

    # Print summary table
    print("\n" + "=" * 60)
    print("  Final Results Summary")
    print("=" * 60)
    print(f"  {'Experiment':<15} {'Best mIoU':<12} {'Best Epoch':<12} {'Params':<10}")
    print(f"  {'─' * 49}")
    for h in histories:
        print(f"  {h['exp_name']:<15} {h['best_miou']:<12.4f} {h['best_epoch']:<12d} {h['params']:<10,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
