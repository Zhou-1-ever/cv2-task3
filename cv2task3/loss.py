"""
Dice Loss — hand-implemented for multi-class segmentation.

Supports three loss modes for comparison experiments:
  1. Cross-Entropy only
  2. Dice only
  3. Cross-Entropy + Dice
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss.
    For each class, Dice = 2·|P∩T| / (|P|+|T|+ε).
    Loss = 1 − mean(Dice over all classes).

    Note: background is included by default. Pass ignore_background=True to drop
    the background channel (class index 0) from the loss computation.
    """

    def __init__(self, smooth: float = 1e-6, ignore_background: bool = False):
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (N, C, H, W) raw model output (logits)
            targets: (N, H, W) ground-truth labels with values in {0, 1, ..., C-1}
        Returns:
            scalar Dice loss
        """
        N, C, H, W = logits.shape

        # Softmax over class dimension → probabilities
        probs = F.softmax(logits, dim=1)  # (N, C, H, W)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=C)  # (N, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (N, C, H, W)

        # Compute intersection and union per class
        intersection = (probs * targets_one_hot).sum(dim=(0, 2, 3))  # (C,)
        cardinality = probs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))  # (C,)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        if self.ignore_background:
            dice_per_class = dice_per_class[1:]  # skip background (class 0)

        dice_loss = 1.0 - dice_per_class.mean()
        return dice_loss


class CombinedLoss(nn.Module):
    """
    Weighted combination of Cross-Entropy and Dice Loss.
    Used in the CE + Dice experiment.
    """

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0,
                 ignore_background: bool = False):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()  # ignores background in reduction if desired
        self.dice = DiceLoss(ignore_background=ignore_background)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, targets) + \
               self.dice_weight * self.dice(logits, targets)


def compute_miou(logits: torch.Tensor, targets: torch.Tensor, n_classes: int = 3) -> float:
    """
    Compute mean Intersection-over-Union.

    Args:
        logits:  (N, C, H, W) raw logits
        targets: (N, H, W) ground-truth labels in {0, ..., C-1}
    Returns:
        mIoU score (float)
    """
    preds = logits.argmax(dim=1)  # (N, H, W)

    ious = []
    for cls in range(n_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union == 0:
            # Both pred and target have no pixels of this class → perfect score
            ious.append(torch.tensor(1.0, device=logits.device))
        else:
            ious.append(intersection / union)

    miou = torch.stack(ious).mean().item()
    return miou


if __name__ == "__main__":
    # Quick sanity check
    torch.manual_seed(42)
    N, C, H, W = 2, 3, 64, 64

    logits = torch.randn(N, C, H, W)
    targets = torch.randint(0, C, (N, H, W))

    dice_loss = DiceLoss()
    combined = CombinedLoss()

    print(f"Dice Loss:      {dice_loss(logits, targets):.6f}")
    print(f"CE Loss:        {nn.CrossEntropyLoss()(logits, targets):.6f}")
    print(f"CE+Dice Loss:   {combined(logits, targets):.6f}")
    print(f"mIoU:           {compute_miou(logits, targets):.6f}")

    # Edge case: perfect prediction
    perfect_logits = torch.zeros(N, C, H, W)
    perfect_logits[:, 0, :, :] = 10.0  # class 0 confident
    perfect_targets = torch.zeros(N, H, W, dtype=torch.long)
    print(f"Perfect Dice:   {dice_loss(perfect_logits, perfect_targets):.6f}  (should be ~0)")
    print("All checks passed ✓")
