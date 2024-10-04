# Copyright (c) OpenMMLab. All rights reserved.
# import os.path as osp
from typing import List, Optional, Tuple, Union
import scipy.io as sio

# import mmcv
import numpy as np
from mmcv.transforms import BaseTransform

# from mmengine.fileio import get_file_backend, list_from_file

from mmagic.registry import TRANSFORMS

# from mmagic.utils import (bbox2mask, brush_stroke_mask, get_irregular_mask,
#                           random_bbox)


@TRANSFORMS.register_module()
class LoadGrokLSTMatFile(BaseTransform):
    """Load a single mat file from corresponding paths. Required
    Keys:
    - [Key]_path

    New Keys:
    - [KEY]
    - ori_[KEY]_shape
    - ori_[KEY]
    - ori_band_name_to_idx_dict if key == "hr_guidance"

    Args:
        key (str): Keys in results to find corresponding path.
        data_field_name (Union[str, List[str]]): The field name of mat file. Default value is "data".
        save_original_mat (bool): Whether save original mat file. Default value is "False".
    """

    def __init__(
        self,
        key: str,  # ['hr_lst', 'lr_lst', 'hr_guidance', 'mask']
        data_field_name: Union[str, List[str]] = "data",
        save_original_mat: bool = False,  # False
    ) -> None:
        self.key = key  # 'mat file'
        self.data_field_name = data_field_name
        self.save_original_mat = save_original_mat

    def transform(self, results: dict) -> dict:
        """Functions to load mat file.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.
        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filenames = results[f"{self.key}_path"]  # mat file path

        if not isinstance(filenames, (List, Tuple)):
            filenames = [str(filenames)]
            is_frames = False
        else:
            filenames = [str(v) for v in filenames]
            is_frames = True

        images = []
        shapes = []
        if self.save_original_mat:
            ori_mats = []

        for filename in filenames:
            if isinstance(self.data_field_name, str):
                data = sio.loadmat(filename)[self.data_field_name]  # H W
            elif isinstance(self.data_field_name, list):
                data = []
                # ori_band_name_to_idx_dict used for NormalizeGrokLSTData!
                ori_band_name_to_idx_dict = dict()
                for idx, field_name in enumerate(self.data_field_name):
                    _data = sio.loadmat(filename)[field_name]
                    ori_band_name_to_idx_dict[f"{field_name}"] = idx
                    data.append(_data)

                results[f"ori_band_name_to_idx_dict"] = ori_band_name_to_idx_dict
                data = np.stack(data, axis=-1)  # h w c
            images.append(data)
            shapes.append(data.shape)
            if self.save_original_mat:
                ori_mats.append(data.copy())

        if not is_frames:
            images = images[0]
            shapes = shapes[0]
            if self.save_original_mat:
                ori_mats = ori_mats[0]

        results[self.key] = images
        results[f"ori_{self.key}_shape"] = shapes
        if self.save_original_mat:
            results[f"ori_{self.key}"] = ori_mats

        return results


# @TRANSFORMS.register_module()
# class LoadImageFromFile(BaseTransform):
#     """Load a single image or image frames from corresponding paths. Required
#     Keys:
#     - [Key]_path

#     New Keys:
#     - [KEY]
#     - ori_[KEY]_shape
#     - ori_[KEY]

#     Args:
#         key (str): Keys in results to find corresponding path.
#         color_type (str): The flag argument for :func:``mmcv.imfrombytes``.
#             Defaults to 'color'.
#         channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
#             Default: 'bgr'.
#         imdecode_backend (str): The image decoding backend type. The backend
#             argument for :func:``mmcv.imfrombytes``.
#             See :func:``mmcv.imfrombytes`` for details.
#             candidates are 'cv2', 'turbojpeg', 'pillow', and 'tifffile'.
#             Defaults to None.
#         use_cache (bool): If True, load all images at once. Default: False.
#         to_float32 (bool): Whether to convert the loaded image to a float32
#             numpy array. If set to False, the loaded image is an uint8 array.
#             Defaults to False.
#         to_y_channel (bool): Whether to convert the loaded image to y channel.
#             Only support 'rgb2ycbcr' and 'rgb2ycbcr'
#             Defaults to False.
#         backend_args (dict, optional): Arguments to instantiate the prefix of
#             uri corresponding backend. Defaults to None.
#     """

#     def __init__(
#         self,
#         key: str,
#         color_type: str = 'color',
#         channel_order: str = 'bgr',
#         imdecode_backend: Optional[str] = None,
#         use_cache: bool = False,
#         to_float32: bool = False,
#         to_y_channel: bool = False,
#         save_original_img: bool = False,
#         backend_args: Optional[dict] = None,
#     ) -> None:

#         self.key = key
#         self.color_type = color_type
#         self.channel_order = channel_order
#         self.imdecode_backend = imdecode_backend
#         self.save_original_img = save_original_img

#         if backend_args is None:
#             # lasy init at loading
#             self.backend_args = None
#             self.file_backend = None
#         else:
#             self.backend_args = backend_args.copy()
#             self.file_backend = get_file_backend(backend_args=backend_args)

#         # cache
#         self.use_cache = use_cache
#         self.cache = dict()

#         # convert
#         self.to_float32 = to_float32
#         self.to_y_channel = to_y_channel

#     def transform(self, results: dict) -> dict:
#         """Functions to load image or frames.

#         Args:
#             results (dict): Result dict from :obj:``mmcv.BaseDataset``.
#         Returns:
#             dict: The dict contains loaded image and meta information.
#         """

#         filenames = results[f'{self.key}_path']

#         if not isinstance(filenames, (List, Tuple)):
#             filenames = [str(filenames)]
#             is_frames = False
#         else:
#             filenames = [str(v) for v in filenames]
#             is_frames = True

#         images = []
#         shapes = []
#         if self.save_original_img:
#             ori_imgs = []

#         for filename in filenames:
#             img = self._load_image(filename)
#             img = self._convert(img)
#             images.append(img)
#             shapes.append(img.shape)
#             if self.save_original_img:
#                 ori_imgs.append(img.copy())

#         if not is_frames:
#             images = images[0]
#             shapes = shapes[0]
#             if self.save_original_img:
#                 ori_imgs = ori_imgs[0]

#         results[self.key] = images
#         results[f'ori_{self.key}_shape'] = shapes
#         results[f'{self.key}_channel_order'] = self.channel_order
#         results[f'{self.key}_color_type'] = self.color_type
#         if self.save_original_img:
#             results[f'ori_{self.key}'] = ori_imgs

#         return results

#     def _load_image(self, filename):
#         """Load an image from file.

#         Args:
#             filename (str): Path of image file.
#         Returns:
#             np.ndarray: Image.
#         """
#         if self.file_backend is None:
#             self.file_backend = get_file_backend(
#                 uri=filename, backend_args=self.backend_args)

#         if (self.backend_args is not None) and (self.backend_args.get(
#                 'backend', None) == 'lmdb'):
#             filename, _ = osp.splitext(osp.basename(filename))

#         if filename in self.cache:
#             img_bytes = self.cache[filename]
#         else:
#             img_bytes = self.file_backend.get(filename)
#             if self.use_cache:
#                 self.cache[filename] = img_bytes

#         img = mmcv.imfrombytes(
#             content=img_bytes,
#             flag=self.color_type,
#             channel_order=self.channel_order,
#             backend=self.imdecode_backend)

#         return img

#     def _convert(self, img: np.ndarray):
#         """Convert an image to the require format.

#         Args:
#             img (np.ndarray): The original image.
#         Returns:
#             np.ndarray: The converted image.
#         """

#         if self.to_y_channel:

#             if self.channel_order.lower() == 'rgb':
#                 img = mmcv.rgb2ycbcr(img, y_only=True)
#             elif self.channel_order.lower() == 'bgr':
#                 img = mmcv.bgr2ycbcr(img, y_only=True)
#             else:
#                 raise ValueError('Currently support only "bgr2ycbcr" or '
#                                  '"bgr2ycbcr".')

#         if img.ndim == 2:
#             img = np.expand_dims(img, axis=2)

#         if self.to_float32:
#             img = img.astype(np.float32)

#         return img

#     def __repr__(self):

#         repr_str = (f'{self.__class__.__name__}('
#                     f'key={self.key}, '
#                     f'color_type={self.color_type}, '
#                     f'channel_order={self.channel_order}, '
#                     f'imdecode_backend={self.imdecode_backend}, '
#                     f'use_cache={self.use_cache}, '
#                     f'to_float32={self.to_float32}, '
#                     f'to_y_channel={self.to_y_channel}, '
#                     f'save_original_img={self.save_original_img}, '
#                     f'backend_args={self.backend_args})')

#         return repr_str
