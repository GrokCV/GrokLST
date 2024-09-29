# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import re
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_file_backend, list_from_file

from mmagic.registry import DATASETS

# IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
#                   '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


@DATASETS.register_module()
class BasicLSTDataset(BaseDataset):
    """BasicLSTDataset for open source projects in OpenMMLab/MMagic.

    This dataset is designed for low-level vision task with image, i.e., super-resolution.
    
    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        data_prefix (dict, optional): Prefix for training data. Defaults to
            dict(img=None, ann=None).
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        filename_tmpl (dict): Template for each filename. Note that the
            template excludes the file extension. Default: dict().
        search_key (str): The key used for searching the folder to get
            data_list. Default: 'gt'.
        backend_args (dict, optional): Arguments to instantiate the prefix of
            uri corresponding backend. Defaults to None.
        suffix (str or tuple[str], optional):  File suffix
            that we are interested in. Default: None.
        recursive (bool): If set to True, recursively scan the
            directory. Default: False.

    Note:

        Assume the file structure as the following:

        .. code-block:: none

            GrokLST (root)
            ├── configs
            ├── data
            │   ├── heihe_lst / large_scale_heihe_lst
            │   │   ├── 30m
            │   │   │   ├── guidance
            |   |   |   ├── lst
            │   │   ├── 60m
            │   │   │   ├── guidance
            |   |   |   ├── lst
            │   │   ├── 120m
            │   │   │   ├── guidance
            |   |   |   ├── lst
            │   │   ├── 240m
            │   │   │   ├── guidance
            |   |   |   ├── lst
            ├── groklst
            ├── tools

    Examples:

        Case 1: Loading heihe_lst or large_scale_heihe_lst dataset for training a GISR model.

        .. code-block:: python

            dataset=dict(
                    type=dataset_type,
                    ann_file="split/trainval.txt",
                    # metainfo=dict(dataset_type=dataset_type, task_name="lst_downscale"),
                    data_root=lst_data_root,
                    data_prefix=dict(
                        lr_lst="60m/lst",
                        hr_lst="30m/lst",
                        hr_guidance="30m/guidance",
                    ),
                    filename_tmpl=dict(),
                    pipeline=train_pipeline)
    """

    METAINFO = dict(dataset_type="BasicLSTDataset", task_name="Downscaling")

    def __init__(
        self,
        ann_file: str = "",
        metainfo: Optional[dict] = None,
        data_root: Optional[str] = None,
        data_prefix: dict = dict(img=""),
        pipeline: List[Union[dict, Callable]] = [],
        test_mode: bool = False,
        filename_tmpl: dict = dict(),
        search_key: Optional[str] = None,
        backend_args: Optional[dict] = None,
        recursive: bool = False,
        **kwards
    ):
        for key in data_prefix:
            if key not in filename_tmpl:
                filename_tmpl[key] = "{}"

        if search_key is None:
            keys = list(data_prefix.keys())
            search_key = keys[0]
        self.search_key = search_key
        self.filename_tmpl = filename_tmpl
        self.use_ann_file = ann_file != ""
        if backend_args is None:
            self.backend_args = None
        else:
            self.backend_args = backend_args.copy()
        # self.img_suffix = img_suffix
        self.recursive = recursive
        self.file_backend = get_file_backend(uri=data_root, backend_args=backend_args)

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwards
        )

    def load_data_list(self) -> List[dict]:
        """Load data list from folder or annotation file.

        Returns:
            list[dict]: A list of annotation.
        """

        path_list = self._get_path_list()  # idx for trainval or test
        
        self.lr_lst = sorted(os.listdir(self.data_prefix["lr_lst"]))
        self.hr_lst = sorted(os.listdir(self.data_prefix["hr_lst"]))
        if "hr_guidance" in self.filename_tmpl.keys():
            self.hr_guidance = sorted(os.listdir(self.data_prefix["hr_guidance"]))
        else:
            self.hr_guidance = None
        # if "lr_mask" in self.filename_tmpl.keys():
        #     self.lr_mask = sorted(os.listdir(self.data_prefix["lr_mask"]))
        # else:
        #     self.lr_mask = None
        # if "hr_mask" in self.filename_tmpl.keys():
        #     self.hr_mask = sorted(os.listdir(self.data_prefix["hr_mask"]))
        # else:
        #     self.hr_mask = None

        assert (len(self.lr_lst) == len(self.hr_lst))
        if self.hr_guidance is not None:
            len(self.lr_lst) == len(self.hr_guidance)
        # if self.lr_mask is not None:
        #     assert len(self.lr_lst) == len(self.lr_mask)
        # if self.hr_mask is not None:
        #     assert len(self.lr_lst) == len(self.hr_mask)
            
        data_list = []
        if self.use_ann_file:
            for idx in path_list:  # idx
                lr_lst = self.lr_lst[int(idx)]
                hr_lst = self.hr_lst[int(idx)]
                if self.hr_guidance is not None:
                    hr_guidance = self.hr_guidance[int(idx)]
                # if self.lr_mask is not None:
                #     lr_mask = self.lr_mask[int(idx)]
                # if self.hr_mask is not None:
                #     hr_mask = self.hr_mask[int(idx)]

                basename, ext = osp.splitext(lr_lst)
                data = dict(key=basename)

                lr_lst_path = osp.join(self.data_prefix["lr_lst"], lr_lst)
                data["lr_lst_path"] = lr_lst_path
                hr_lst_path = osp.join(self.data_prefix["hr_lst"], hr_lst)
                data["hr_lst_path"] = hr_lst_path
                if self.hr_guidance is not None:
                    hr_guidance_path = osp.join(self.data_prefix["hr_guidance"], hr_guidance)
                    data["hr_guidance_path"] = hr_guidance_path
                else: 
                    data["hr_guidance_path"] = None
                # if self.lr_mask is not None:
                #     data["lr_mask_path"] = osp.join(self.data_prefix["lr_mask"], lr_mask)
                # else:
                #     data["lr_mask_path"] = None
                # if self.hr_mask is not None:
                #     data["hr_mask_path"] = osp.join(self.data_prefix["hr_mask"], hr_mask)
                # else:
                #     data["hr_mask_path"] = None
                data_list.append(data)
        else: # todo 实现非下标来获取数据，即使用文件名的方式来获取数据
            raise NotImplementedError

        return data_list

    def _get_path_list(self):
        """Get list of paths from annotation file or folder of dataset.

        Returns:
            list[dict]: A list of paths.
        """

        path_list = []
        if self.use_ann_file:
            path_list = self._get_path_list_from_ann()
        else:
            path_list = self._get_path_list_from_folder()

        return path_list

    def _get_path_list_from_ann(self):
        """Get list of paths from annotation file.

        Returns:
            List: List of paths.
        """

        ann_list = list_from_file(self.ann_file, backend_args=self.backend_args)
        path_list = []
        for ann in ann_list:
            if ann.isspace() or ann == "":
                continue
            path = ann.split(" ")[0]
            # Compatible with Windows file systems
            path = path.replace("/", os.sep)
            path_list.append(path)

        return path_list

    def _get_path_list_from_folder(self):
        """Get list of paths from folder.

        Returns:
            List: List of paths.
        """

        path_list = []
        folder = self.data_prefix[self.search_key]
        tmpl = self.filename_tmpl[self.search_key].format("")
        virtual_path = self.filename_tmpl[self.search_key].format(".*")
        for img_path in self.file_backend.list_dir_or_file(
            dir_path=folder,
            list_dir=False,
            # suffix=self.img_suffix,
            recursive=self.recursive,
        ):
            basename, ext = osp.splitext(img_path)
            if re.match(virtual_path, basename):
                img_path = img_path.replace(tmpl + ext, ext)
                path_list.append(img_path)

        return path_list
