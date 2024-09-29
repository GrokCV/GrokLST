# Copyright (c) OpenMMLab. All rights reserved.
"""Evaluation metrics based on pixels."""

from mmagic.registry import METRICS
from mmagic.evaluation.metrics.base_sample_wise_metric import BaseSampleWiseMetric


@METRICS.register_module()
class BIAS(BaseSampleWiseMetric):
    """Statistical indicator: Bias for LST image.

    mean(a-b)

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
        - BIAS (float): mean(a-b)
    """

    metric = "BIAS"

    def process_image(self, gt, pred, mask=None):
        """Process an image.

        Args:
            gt (Torch | np.ndarray): GT image.
            pred (Torch | np.ndarray): Pred image.
            mask (Torch | np.ndarray): Mask of evaluation.
        Returns:
            result (np.ndarray): BIAS result.
        """

        # mean(a-b)
        # diff = gt - pred
        diff = pred - gt

        if self.mask_key is not None:
            diff *= mask
            result = diff.sum() / (mask.sum() + 1e-12)
        else:
            result = diff.mean()

        return result
