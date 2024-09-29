# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmagic.registry import MODELS
from .loss_wrapper import masked_loss

_reduction_modes = ["none", "mean", "sum"]


@masked_loss
def smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth L1 loss. https://www.cnblogs.com/wangguchangqing/p/12021638.html

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated Smooth L1 loss.
    """
    t = torch.abs(pred - target)
    ret = torch.where(t < 1, 0.5 * t**2, t - 0.5)

    return torch.mean(ret)


@MODELS.register_module()
class SmoothL1Loss(nn.Module):
    """SmoothL1Loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduce loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self, loss_weight: float = 1.0, reduction: str = "mean", sample_wise: bool = False) -> None:
        super().__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Unsupported reduction mode: {reduction}. " f"Supported ones are: {_reduction_modes}")

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * smooth_l1_loss(
            pred, target, weight, reduction=self.reduction, sample_wise=self.sample_wise
        )
