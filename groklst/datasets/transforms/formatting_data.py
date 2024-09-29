# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

from mmcv.transforms.base import BaseTransform
from mmcv.transforms import to_tensor
from mmagic.registry import TRANSFORMS
from mmagic.structures import DataSample
from mmagic.utils import all_to_tensor
import torch


@TRANSFORMS.register_module()
class PackLSTInputs(BaseTransform):
    """Pack data into DataSample for training, evaluation and testing.

    MMagic follows the design of data structure from MMEngine.
        Data from the loader will be packed into data field of DataSample.
        More details of DataSample refer to the documentation of MMEngine:
        https://mmengine.readthedocs.io/en/latest/advanced_tutorials/data_element.html

    Args:
        keys Tuple[List[str], str, None]: The keys to saved in returned
            inputs, which are used as the input of models, default to
            ['img', 'noise', 'merged'].
        data_keys Tuple[List[str], str, None]: The keys to saved in
            `data_field` of the `data_samples`.
        meta_keys Tuple[List[str], str, None]: The meta keys to saved
            in `metainfo` of the `data_samples`. All the other data will
            be packed into the data of the `data_samples`
    """

    DATA_KEYS_MAPPING = {"hr_lst": "gt_img"}

    def __init__(
        self,
        keys: Tuple[List[str], str] = [],  # not use
        meta_keys: Tuple[List[str], str] = [
            "lr_lst_path",
            "hr_lst_path",
            "hr_guidance_path",
            "lr_mask_path",
            "hr_mask_path",
            "ori_mask_shape",
            "ori_lr_lst_shape",
            "ori_hr_lst_shape",
            "ori_hr_guidance_shape",
        ],
        # data_keys: Tuple[List[str], str] = ["hr_lst", "lr_mask", "hr_mask"],
        data_keys: Tuple[List[str], str] = ["hr_lst"],
    ) -> None:
        assert keys is not None, "keys in PackInputs can not be None."
        assert data_keys is not None, "data_keys in PackInputs can not be None."
        assert meta_keys is not None, "meta_keys in PackInputs can not be None."

        self.keys = keys if isinstance(keys, List) else [keys]
        self.data_keys = data_keys if isinstance(data_keys, List) else [data_keys]
        self.meta_keys = meta_keys if isinstance(meta_keys, List) else [meta_keys]

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (obj:`dict`): The forward data of models.
              According to different tasks, the `inputs` may contain images,
              videos, labels, text, etc.

            - 'data_samples' (obj:`DataSample`): The annotation info of the
                sample.
        """

        # prepare inputs
        inputs_keys = ["lr_lst"]
        inputs = dict()
        for k in inputs_keys:
            value = results.get(k, None)
            if value is not None:
                inputs[k] = all_to_tensor(value) # H W C -> C H W

        # return the inputs as tensor, if it has only one item
        if len(inputs.values()) == 1:
            inputs = list(inputs.values())[0]

        # prepare hr_guidance same as inputs
        hr_guidance_keys = ["hr_guidance"]
        hr_guidance = dict()
        for k in hr_guidance_keys:
            value = results.get(k, None)
            if value is not None:
                hr_guidance[k] = all_to_tensor(value)

        # return the hr_guidance as tensor, if it has only one item
        if len(hr_guidance.values()) == 1:
            hr_guidance = list(hr_guidance.values())[0]
        
        # # prepare gui_mask
        if results.get("gui_mask", None) is not None:
            value = results["gui_mask"]
            gui_mask = to_tensor(value)

        # prepare masks
        masks_keys = ["lr_mask", "hr_mask"]
        masks = dict()
        for k in masks_keys:
            value = results.get(k, None)
            if value is not None:
                masks[k] = all_to_tensor(value)

        data_sample = DataSample()
        # prepare metainfo and data in DataSample according to predefined keys
        predefined_data = {k: v for (k, v) in results.items() if k not in (self.data_keys + self.meta_keys)}
        data_sample.set_predefined_data(predefined_data)  # only setting sample_idx

        # set masks data
        if len(masks) > 0:
            data_sample.set_tensor_data(masks)

        # prepare metainfo in DataSample according to user-provided meta_keys
        required_metainfo = {k: v for (k, v) in results.items() if k in self.meta_keys}
        data_sample.set_metainfo(required_metainfo)

        # prepare metainfo in DataSample according to user-provided data_keys
        required_data = {
            self.DATA_KEYS_MAPPING[k]: v  # hr_lst -> gt_img for computing loss
            for (k, v) in results.items()
            if k in self.data_keys
        }
        # required_data[] = v
        data_sample.set_tensor_data(required_data) # gt_img

        if len(hr_guidance) == 0:  # for single modality
            return {"inputs": inputs, "data_samples": data_sample}
        else:  # for two modality
            if results.get("gui_mask", None) is not None:
                return {
                    "inputs": inputs,
                    "hr_guidance": hr_guidance,
                    "gui_mask": gui_mask,
                    "data_samples": data_sample,
                }
            else:
                return {"inputs": inputs, "hr_guidance": hr_guidance, "data_samples": data_sample}

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__

        return repr_str


@TRANSFORMS.register_module()
class PackOpticalSARInputs(BaseTransform):
    """Pack data into DataSample for training, evaluation and testing.

    MMagic follows the design of data structure from MMEngine.
        Data from the loader will be packed into data field of DataSample.
        More details of DataSample refer to the documentation of MMEngine:
        https://mmengine.readthedocs.io/en/latest/advanced_tutorials/data_element.html

    Args:
        keys Tuple[List[str], str, None]: The keys to saved in returned
            inputs, which are used as the input of models, default to
            ['img', 'noise', 'merged'].
        data_keys Tuple[List[str], str, None]: The keys to saved in
            `data_field` of the `data_samples`.
        meta_keys Tuple[List[str], str, None]: The meta keys to saved
            in `metainfo` of the `data_samples`. All the other data will
            be packed into the data of the `data_samples`
    """

    def __init__(
        self,
        keys: Tuple[List[str], str] = ['img', 'sar_vh'], # 'sar_vv'
        meta_keys: Tuple[List[str], str] = [], # todo
        data_keys: Tuple[List[str], str] = ['gt'],
    ) -> None:

        assert keys is not None, \
            'keys in PackInputs can not be None.'
        assert data_keys is not None, \
            'data_keys in PackInputs can not be None.'
        assert meta_keys is not None, \
            'meta_keys in PackInputs can not be None.'

        self.keys = keys if isinstance(keys, List) else [keys]
        self.data_keys = data_keys if isinstance(data_keys,
                                                 List) else [data_keys]
        self.meta_keys = meta_keys if isinstance(meta_keys,
                                                 List) else [meta_keys]

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (obj:`dict`): The forward data of models.
              According to different tasks, the `inputs` may contain images,
              videos, labels, text, etc.

            - 'data_samples' (obj:`DataSample`): The annotation info of the
                sample.
        """

        # prepare inputs
        inputs = dict()
        sar_tensor_list = []
        for k in self.keys:
            value = results.get(k, None)
            if 'sar' in k:
                sar_tensor_list.append(all_to_tensor(value))
                continue
            if value is not None:
                inputs[k] = all_to_tensor(value)

        # generate "sar" field for inputs.
        all_sar_tensor = torch.cat(sar_tensor_list, dim=0) # C H W
        inputs['sar'] = all_sar_tensor

        # return the inputs as tensor, if it has only one item
        if len(inputs.values()) == 1:
            inputs = list(inputs.values())[0]

        data_sample = DataSample()
        # prepare metainfo and data in DataSample according to predefined keys
        predefined_data = {
            k: v
            for (k, v) in results.items()
            if k not in (self.data_keys + self.meta_keys)
        }
        data_sample.set_predefined_data(predefined_data)

        # prepare metainfo in DataSample according to user-provided meta_keys
        required_metainfo = {
            k: v
            for (k, v) in results.items() if k in self.meta_keys
        }
        data_sample.set_metainfo(required_metainfo)

        # prepare metainfo in DataSample according to user-provided data_keys
        required_data = {
            k: v
            for (k, v) in results.items() if k in self.data_keys
        }
        data_sample.set_tensor_data(required_data)

        return {'inputs': inputs, 'data_samples': data_sample}

    def __repr__(self) -> str:

        repr_str = self.__class__.__name__

        return repr_str


# @TRANSFORMS.register_module()
# class PackInputs(BaseTransform):
#     """Pack data into DataSample for training, evaluation and testing.

#     MMagic follows the design of data structure from MMEngine.
#         Data from the loader will be packed into data field of DataSample.
#         More details of DataSample refer to the documentation of MMEngine:
#         https://mmengine.readthedocs.io/en/latest/advanced_tutorials/data_element.html

#     Args:
#         keys Tuple[List[str], str, None]: The keys to saved in returned
#             inputs, which are used as the input of models, default to
#             ['img', 'noise', 'merged'].
#         data_keys Tuple[List[str], str, None]: The keys to saved in
#             `data_field` of the `data_samples`.
#         meta_keys Tuple[List[str], str, None]: The meta keys to saved
#             in `metainfo` of the `data_samples`. All the other data will
#             be packed into the data of the `data_samples`
#     """

#     def __init__(
#         self,
#         keys: Tuple[List[str], str] = ['merged', 'img'],
#         meta_keys: Tuple[List[str], str] = [],
#         data_keys: Tuple[List[str], str] = [],
#     ) -> None:

#         assert keys is not None, \
#             'keys in PackInputs can not be None.'
#         assert data_keys is not None, \
#             'data_keys in PackInputs can not be None.'
#         assert meta_keys is not None, \
#             'meta_keys in PackInputs can not be None.'

#         self.keys = keys if isinstance(keys, List) else [keys]
#         self.data_keys = data_keys if isinstance(data_keys,
#                                                  List) else [data_keys]
#         self.meta_keys = meta_keys if isinstance(meta_keys,
#                                                  List) else [meta_keys]

#     def transform(self, results: dict) -> dict:
#         """Method to pack the input data.

#         Args:
#             results (dict): Result dict from the data pipeline.

#         Returns:
#             dict: A dict contains

#             - 'inputs' (obj:`dict`): The forward data of models.
#               According to different tasks, the `inputs` may contain images,
#               videos, labels, text, etc.

#             - 'data_samples' (obj:`DataSample`): The annotation info of the
#                 sample.
#         """

#         # prepare inputs
#         inputs = dict()
#         for k in self.keys:
#             value = results.get(k, None)
#             if value is not None:
#                 inputs[k] = all_to_tensor(value)

#         # return the inputs as tensor, if it has only one item
#         if len(inputs.values()) == 1:
#             inputs = list(inputs.values())[0]

#         data_sample = DataSample()
#         # prepare metainfo and data in DataSample according to predefined keys
#         predefined_data = {
#             k: v
#             for (k, v) in results.items()
#             if k not in (self.data_keys + self.meta_keys)
#         }
#         data_sample.set_predefined_data(predefined_data)

#         # prepare metainfo in DataSample according to user-provided meta_keys
#         required_metainfo = {
#             k: v
#             for (k, v) in results.items() if k in self.meta_keys
#         }
#         data_sample.set_metainfo(required_metainfo)

#         # prepare metainfo in DataSample according to user-provided data_keys
#         required_data = {
#             k: v
#             for (k, v) in results.items() if k in self.data_keys
#         }
#         data_sample.set_tensor_data(required_data)

#         return {'inputs': inputs, 'data_samples': data_sample}

#     def __repr__(self) -> str:

#         repr_str = self.__class__.__name__

#         return repr_str
