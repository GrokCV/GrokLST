# Copyright (c) OpenMMLab. All rights reserved.
"""Evaluation metrics based on pixels."""

import numpy as np

from mmagic.registry import METRICS
from mmagic.evaluation.metrics.base_sample_wise_metric import BaseSampleWiseMetric


@METRICS.register_module()
class LSTMAE(BaseSampleWiseMetric):
    """Mean Absolute Error metric for LST image.
    Note that the main difference between LSTMAE and default MAE of mmagic is that LSTMAE do not normlize gt and pred.
    i.e., LSTMAE=abs(gt-pred), MAE=abs((gt/255.)-(pred/255.))

    mean(abs(a-b))

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
        - MAE (float): Mean of Absolute Error
    """

    metric = "MAE"

    def process_image(self, gt, pred, mask=None):
        """Process an image.

        Args:
            gt (Tensor | np.ndarray): GT image.
            pred (Tensor | np.ndarray): Pred image.
            mask (Tensor | np.ndarray): Mask of evaluation.
        Returns:
            result (np.ndarray): MAE result.
        """

        diff = gt - pred
        diff = abs(diff)
        # print(f"min={pred.min()}, max={pred.max()}")

        if self.mask_key is not None:
            diff *= mask  # broadcast for channel dimension
            scale = np.prod(diff.shape) / np.prod(mask.shape)
            result = diff.sum() / (mask.sum() * scale + 1e-12)
        else:
            result = diff.mean()

        return result
