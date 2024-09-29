# Copyright (c) OpenMMLab. All rights reserved.
"""Evaluation metrics based on pixels."""

from mmagic.registry import METRICS
from mmagic.evaluation.metrics.base_sample_wise_metric import BaseSampleWiseMetric
import numpy as np


@METRICS.register_module()
class CC(BaseSampleWiseMetric):
    """Correlation Coefficient(CC) for LST image.

    # sqrt(mean((a-b)^2))

    Args:

        gt_key (str): Key of ground-truth. Default: 'gt_img'
        pred_key (str): Key of prediction. Default: 'pred_img'
        mask_key (str, optional): Key of mask, if mask_key is None, calculate
            all regions. Default: None
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None

    Metrics:
        - CC (float): Correlation Coefficient
    """

    metric = "CC"

    def process_image(self, gt, pred, mask=None):
        """Process an image.

        Args:
            gt (Torch | np.ndarray): GT image.
            pred (Torch | np.ndarray): Pred image.
            mask (Torch | np.ndarray): Mask of evaluation.
        Returns:
            result (np.ndarray): Correlation Coefficient result.
        """

        if self.mask_key is not None:
            # n = mask.sum()
            # # if n == 0:
            # #     n = 1e-12
            # gt = gt.reshape(-1)[mask.reshape(-1)]  # to 1d vector
            # pred = pred.reshape(-1)[mask.reshape(-1)]  # # to 1d vector
            # gt_mean = gt.sum() / n
            # pred_mean = pred.sum() / n
            # gt_mean_square = ((gt - gt_mean) ** 2).sum() / n
            # pred_mean_square = ((pred - pred_mean) ** 2).sum() / n
            # result = (((gt - gt_mean) * (pred - pred_mean)).sum() / n) / (
            #     (np.sqrt(gt_mean_square)) * (np.sqrt(pred_mean_square)) + 1e-12
            # )
            # # opt-2
            n = mask.sum()
            gt = gt * mask
            pred = pred * mask
            gt_mean = gt.sum() / n
            pred_mean = pred.sum() / n
            gt[mask == 0] = gt_mean
            pred[mask == 0] = pred_mean
            gt_mean_square = ((gt - gt_mean) ** 2).sum() / n
            pred_mean_square = ((pred - pred_mean) ** 2).sum() / n
            result = (((gt - gt_mean) * (pred - pred_mean)).sum() / n) / (
                (np.sqrt(gt_mean_square)) * (np.sqrt(pred_mean_square)) + 1e-12
            )
        else:
            gt_mean = gt.mean()
            pred_mean = pred.mean()
            gt_mean_square = ((gt - gt_mean) ** 2).mean()
            pred_mean_square = ((pred - pred_mean) ** 2).mean()
            result = ((gt - gt_mean) * (pred - pred_mean)).mean() / (
                (np.sqrt(gt_mean_square)) * (np.sqrt(pred_mean_square)) + 1e-12
            )

        return result
