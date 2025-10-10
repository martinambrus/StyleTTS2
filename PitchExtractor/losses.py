"""Loss builders for training the pitch extraction model."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class BCEWithLogitsLossWithSmoothing(nn.Module):
    """Binary cross entropy with optional label smoothing."""

    def __init__(self, smoothing: float = 0.0, reduction: str = "mean") -> None:
        super().__init__()
        self.smoothing = float(max(0.0, min(1.0, smoothing)))
        self.reduction = reduction

    def _smooth_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if self.smoothing <= 0.0:
            return targets
        smooth = self.smoothing * 0.5
        return targets * (1.0 - self.smoothing) + smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        smoothed_targets = self._smooth_targets(targets)
        return F.binary_cross_entropy_with_logits(inputs, smoothed_targets, reduction=self.reduction)


class SigmoidFocalLossWithLogits(nn.Module):
    """Focal loss operating on logits with optional label smoothing."""

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)
        self.reduction = reduction
        self.label_smoothing = float(max(0.0, min(1.0, label_smoothing)))

    def _smooth_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing <= 0.0:
            return targets
        smooth = self.label_smoothing * 0.5
        return targets * (1.0 - self.label_smoothing) + smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        targets = self._smooth_targets(targets)
        prob = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        focal_weight = (1.0 - p_t).clamp_min(0.0) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal_weight = focal_weight * alpha_t
        loss = ce_loss * focal_weight
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_silence_loss(loss_config: dict) -> nn.Module:
    """Construct the silence loss according to configuration options."""

    smoothing = float(loss_config.get("silence_label_smoothing", 0.0))
    reduction = str(loss_config.get("silence_loss_reduction", "mean"))
    use_focal = bool(loss_config.get("silence_use_focal_loss", False))

    if use_focal:
        alpha = loss_config.get("silence_focal_alpha")
        if alpha is not None:
            alpha = float(alpha)
        gamma = float(loss_config.get("silence_focal_gamma", 2.0))
        return SigmoidFocalLossWithLogits(
            gamma=gamma,
            alpha=alpha,
            reduction=reduction,
            label_smoothing=smoothing,
        )

    return BCEWithLogitsLossWithSmoothing(smoothing=smoothing, reduction=reduction)
