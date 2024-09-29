# Copyright (c) OpenMMLab. All rights reserved.
# import os.path as osp
from typing import List, Optional, Tuple
import scipy.io as sio
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmagic.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PadBands(BaseTransform):
    """Pads the channel dimension of the input array to the target number of channels.

    New Keys: None

    Args:
        key (str): The key of results, and the results[key] to be normalized.
        pad_mode (str): supports all modes in numpy.pad, such as "constant", "reflect", "symmetric", "edge",...
        pad_value (float): if pad_mode is "constant", use pad_value.
        kwargs (dict): other args used in numpy.pad. 
    """

    def __init__(
        self,
        key: str = "hr_guidance",
        pad_mode: str = "constant",
        pad_value: float = 0.0,
        kwargs: dict = {},
    ) -> None:
        self.key = key
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        self.kwargs = kwargs

    def transform(self, results: dict) -> dict:
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        data = results[self.key]
        H, W, C = data.shape
        ori_band_name_to_idx_dict = results.get('ori_band_name_to_idx_dict', None)
        target_channels = len(ori_band_name_to_idx_dict)
        if C > target_channels:
            raise ValueError("Target channels must be greater than current channels.")
        elif C == target_channels:
            return results
        
        # Calculate padding sizes
        pad_before = (target_channels - C) // 2
        pad_after = target_channels - C - pad_before
        
        # Pad the array
        pad_width = ((0, 0), (0, 0), (pad_before, pad_after))
        if self.pad_mode == "constant":
            data = np.pad(data, pad_width, mode=self.pad_mode, constant_values=self.pad_value, **self.kwargs)
        else:
            data = np.pad(data, pad_width, mode=self.pad_mode, **self.kwargs)
            
        results[self.key] = data

        return results
