import numpy as np
from mmcv.transforms import BaseTransform
from mmagic.registry import TRANSFORMS
import random
import math


@TRANSFORMS.register_module()
class RandomDropBands(BaseTransform):
    """Randomly drop bands of data with the given bands_loss_rate_dict.

    New Keys:
    - gui_mask
    - with_zero_padding
    - saved_band_name_to_idx_dict

    Args:
        key (str): The key of results, and the bands of the results[key] to be dropped.
        seed (int): Random seed.
        bands_loss_rate_dict (dict): Band loss dictionary, including every band name and its channel index and loss rate. For example, 'dem': (0, 0.5) means that the 'dem' band corresponds to the 0th channel of the guidance tensor (B, H, W, C=10), i.e., guidance[B, H, W, 0], and its band loss rate is 0.5.
        drop_band (bool): Whether drop bands.
        with_zero_padding (bool): Zero pad in-place on the discarded bands.
        min_retained_band_count (int): Since all bands might be discarded simultaneously, we hope to retain at least a "min_retained_band_count" of bands to serve as guide information.

    """

    def __init__(
        self,
        key: str = "hr_guidance",
        seed: int = None,
        bands_loss_rate_dict: dict = dict(),
        drop_band: bool = True,
        with_zero_padding: bool = False,
        min_retained_band_count: int = 1,
    ) -> None:
        self.key = key
        if seed is not None:
            assert type(seed) == int, "seed should be the type of 'int'."
            random.seed(seed)
        self.bands_loss_rate_dict = bands_loss_rate_dict
        self.drop_band = drop_band
        self.with_zero_padding = with_zero_padding
        self.min_retained_band_count = min_retained_band_count

    def transform(self, results: dict) -> dict:
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        results['drop_band'] = self.drop_band
        if self.drop_band:
            data = results[self.key]  # H, W, C
            H, W, C = data.shape
            ori_band_name_to_idx_dict = results['ori_band_name_to_idx_dict']
            assert set(ori_band_name_to_idx_dict.keys()).issubset(self.bands_loss_rate_dict.keys())

            # 记录波段是否丢失, 若为 0 则表示丢失, 反之, 1 表示保留
            gui_mask = [1.0 for _ in range(C)]  # 默认全部保留
            saved_bands = []
            saved_band_name_to_idx_dict = {}
            idx = 0
            for band_name, channel_idx in ori_band_name_to_idx_dict.items():
                drop_ratio = self.bands_loss_rate_dict[band_name]
                random_probability = random.random()
                channel_idx = int(ori_band_name_to_idx_dict[band_name])

                if random_probability < drop_ratio:  # drop band
                    if self.with_zero_padding:
                        data[..., channel_idx] = 0
                        gui_mask[channel_idx] = 0
                else:
                    _band = data[..., channel_idx]
                    saved_bands.append(_band)
                    saved_band_name_to_idx_dict[f'{band_name}'] = idx
                    idx += 1

            if not self.with_zero_padding:
                # Avoid the "ValueError('need at least one array to stack')"  \
                # caused by discarding all bands!
                _idx = 0
                if len(saved_bands) == 0 and len(saved_band_name_to_idx_dict) == 0: 
                    random_band_name_list = random.sample(list(ori_band_name_to_idx_dict.keys()), self.min_retained_band_count)
                    for band_name in random_band_name_list:
                        _channel_idx = ori_band_name_to_idx_dict[band_name]
                        _band = data[..., _channel_idx]
                        saved_bands.append(_band)
                        saved_band_name_to_idx_dict[band_name] = _idx
                        _idx += 1

                data = np.stack(saved_bands, axis=-1) # H W C
                gui_mask = [0.0 for _ in range(C)] # 默认全部丢弃，方便处理
                for band_name, _idx in saved_band_name_to_idx_dict.items():
                    ori_idx = int(ori_band_name_to_idx_dict[band_name])
                    gui_mask[ori_idx] = 1.0
                results["gui_mask"] = gui_mask
            else:
                gui_mask = np.array(gui_mask).astype(np.float32)
                results["gui_mask"] = gui_mask
                
            # print(f"Data shape before assertion: {data.shape}")
            assert len(data.shape) == 3, "len(data.shape) is not equal 3!"
            results['saved_band_name_to_idx_dict'] = saved_band_name_to_idx_dict
            results[self.key] = data
            results['with_zero_padding'] = self.with_zero_padding

            return results
        else:
            return results
