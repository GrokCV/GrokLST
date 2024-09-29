# Copyright (c) OpenMMLab. All rights reserved.
import math
from logging import WARNING
from typing import List, Optional, Sequence, Tuple, Union

import torch
# import torch.nn.functional as F
# from mmengine import print_log
# from mmengine.model import ImgDataPreprocessor
from mmengine.utils import is_seq_of
from torch import Tensor

from mmagic.registry import MODELS
from mmagic.structures import DataSample
from mmagic.utils.typing import SampleList
from mmagic.models.data_preprocessors import DataPreprocessor

CastData = Union[tuple, dict, DataSample, Tensor, list]


@MODELS.register_module()
class LSTDataPreprocessor(DataPreprocessor):
    """LSTDataPreprocessor is same as DataPreprocessor, just override "destruct" function,
      i.e., delete _batch_outputs = _batch_outputs.clamp_(0, 255).

    LST pre-processor for generative models. This class provide
    normalization and bgr to rgb conversion for image tensor inputs. The input
    of this classes should be dict which keys are `inputs` and `data_samples`.

    Besides to process tensor `inputs`, this class support dict as `inputs`.
    - If the value is `Tensor` and the corresponding key is not contained in
    :attr:`_NON_IMAGE_KEYS`, it will be processed as image tensor.
    - If the value is `Tensor` and the corresponding key belongs to
    :attr:`_NON_IMAGE_KEYS`, it will not remains unchanged.
    - If value is string or integer, it will not remains unchanged.

    Args:
        mean (Sequence[float or int], float or int, optional): The pixel mean
            of image channels. Noted that normalization operation is performed
            *after channel order conversion*. If it is not specified, images
            will not be normalized. Defaults None.
        std (Sequence[float or int], float or int, optional): The pixel
            standard deviation of image channels. Noted that normalization
            operation is performed *after channel order conversion*. If it is
            not specified, images will not be normalized. Defaults None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        pad_mode (str): Padding mode for ``torch.nn.functional.pad``.
            Defaults to 'constant'.
        non_image_keys (List[str] or str): Keys for fields that not need to be
            processed (padding, channel conversion and normalization) as
            images. If not passed, the keys in :attr:`_NON_IMAGE_KEYS` will be
            used. This argument will only work when `inputs` is dict or list
            of dict. Defaults to None.
        non_concatenate_keys (List[str] or str): Keys for fields that not need
            to be concatenated. If not passed, the keys in
            :attr:`_NON_CONCATENATE_KEYS` will be used. This argument will only
            work when `inputs` is dict or list of dict. Defaults to None.
        output_channel_order (str, optional): The desired image channel order
            of output the data preprocessor. This is also the desired input
            channel order of model (and this most likely to be the output
            order of model). If not passed, no channel order conversion will
            be performed. Defaults to None.
        data_keys (List[str] or str): Keys to preprocess in data samples.
            Defaults to 'gt_img'.
        input_view (tuple, optional): The view of input tensor. This
            argument maybe deleted in the future. Defaults to None.
        output_view (tuple, optional): The view of output tensor. This
            argument maybe deleted in the future. Defaults to None.
        stack_data_sample (bool): Whether stack a list of data samples to one
            data sample. Only support with input data samples are
            `DataSamples`. Defaults to True.
    """

    def __init__(
        self,
        mean: Union[Sequence[Union[float, int]], float, int] = None,
        std: Union[Sequence[Union[float, int]], float, int] = None,
        pad_size_divisor: int = 1,
        pad_value: Union[float, int] = 0,
        pad_mode: str = "constant",
        non_image_keys: Optional[Tuple[str, List[str]]] = None,
        non_concentate_keys: Optional[Tuple[str, List[str]]] = None,
        output_channel_order: Optional[str] = None,
        # you can set keys in data samples, e.g., "gt_img", for preprocessing,
        # but we do not norm gt_img field in data samples!
        data_keys: Union[List[str], str] = None,
        input_view: Optional[tuple] = None,
        output_view: Optional[tuple] = None,
        stack_data_sample=True,
    ):
        super().__init__(
            mean,
            std,
            pad_size_divisor,
            pad_value,
            pad_mode,
            non_image_keys,
            non_concentate_keys,
            output_channel_order,
            data_keys,
            input_view,
            output_view,
            stack_data_sample,
        )
        self.mean = mean
        self.std = std
        self._enable_normalize = False
        if self.mean is not None and self.std is not None:
            self._enable_normalize = True

    def forward(self, data: dict, training: bool = False) -> dict:
        """Performs normalization、padding and channel order conversion.

        Args:
            data (dict): Input data to process.
            training (bool): Whether to in training mode. Default: False.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)
        _batch_inputs = data['inputs']
        _batch_data_samples = data.get('data_samples', None)

        # process input
        if isinstance(_batch_inputs, torch.Tensor):
            _batch_inputs, _batch_data_samples = \
                self._preprocess_image_tensor(
                    _batch_inputs, _batch_data_samples)
        elif is_seq_of(_batch_inputs, torch.Tensor):
            _batch_inputs, _batch_data_samples = \
                self._preprocess_image_list(
                    _batch_inputs, _batch_data_samples)
        elif isinstance(_batch_inputs, dict):
            _batch_inputs, _batch_data_samples = \
                self._preprocess_dict_inputs(
                    _batch_inputs, _batch_data_samples)
        elif is_seq_of(_batch_inputs, dict):
            # convert list of dict to dict of list
            keys = _batch_inputs[0].keys()
            dict_input = {k: [inp[k] for inp in _batch_inputs] for k in keys}
            _batch_inputs, _batch_data_samples = \
                self._preprocess_dict_inputs(
                    dict_input, _batch_data_samples)
        else:
            raise ValueError('Only support following inputs types: '
                             '\'torch.Tensor\', \'List[torch.Tensor]\', '
                             '\'dict\', \'List[dict]\'. But receive '
                             f'\'{type(_batch_inputs)}\'.')
        data['inputs'] = _batch_inputs

        # process data samples
        if _batch_data_samples:
            _batch_data_samples = self._preprocess_data_sample(
                _batch_data_samples, training)

        data['data_samples'] = _batch_data_samples
        _batch_hr_guidance = data.get('hr_guidance',None)
        if _batch_hr_guidance is not None: 
            _batch_hr_guidance = torch.stack(_batch_hr_guidance)
            data['hr_guidance'] = _batch_hr_guidance

        if data.get('gui_mask', None) is not None:
            _gui_mask = data['gui_mask']
            _gui_mask = torch.stack(_gui_mask)
            data['gui_mask'] = _gui_mask

        return data
    
    # here, we delete _batch_outputs = _batch_outputs.clamp_(0, 255),
    # because LST data does not need to clamp to 0-255.
    def destruct(
        self, outputs: Tensor, data_samples: Union[SampleList, DataSample, None] = None, key: str = "img"
    ) -> Union[list, Tensor]:
        """Destruct padding, normalization and convert channel order to BGR if
        could. If `data_samples` is a list, outputs will be destructed as a
        batch of tensor. If `data_samples` is a `DataSample`, `outputs` will be
        destructed as a single tensor.

        Before feed model outputs to visualizer and evaluator, users should
        call this function for model outputs and inputs.

        Use cases:

        >>> # destruct model outputs.
        >>> # model outputs share the same preprocess information with inputs
        >>> # ('img') therefore use 'img' as key
        >>> feats = self.forward_tensor(inputs, data_samples, **kwargs)
        >>> feats = self.data_preprocessor.destruct(feats, data_samples, 'img')

        >>> # destruct model inputs for visualization
        >>> for idx, data_sample in enumerate(data_samples):
        >>>     destructed_input = self.data_preprocessor.destruct(
        >>>         inputs[idx], data_sample, key='img')
        >>>     data_sample.set_data({'input': destructed_input})

        Args:
            outputs (Tensor): Tensor to destruct.
            data_samples (Union[SampleList, DataSample], optional): Data
                samples (or data sample) corresponding to `outputs`.
                Defaults to None
            key (str): The key of field in data sample. Defaults to 'img'.

        Returns:
            Union[list, Tensor]: Destructed outputs.
        """
        # NOTE: only support passing tensor sample, if the output of model is
        # a dict, users should call this manually.
        # Since we do not know whether the outputs is image tensor.
        _batch_outputs = self._destruct_norm_and_conversion(outputs, data_samples, key)
        _batch_outputs = self._destruct_padding(_batch_outputs, data_samples)
        # _batch_outputs = _batch_outputs.clamp_(0, 255)
        return _batch_outputs
