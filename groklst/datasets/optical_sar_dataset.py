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
class OpticalSARDataset(BaseDataset):
    """OpticalSARDataset for open source projects in OpenMMLab/MMagic.
    光学SAR融合去云赛道:    http://rsipac.whu.edu.cn/subject_one

    数据集请在百度云下载链接: https://pan.baidu.com/s/1PC6g229Y8TN-29C4Vu_hbw?pwd=1111 提取码: 1111
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
            │   ├── heihe_lst / large_scale_heihe_lstDataset (Optical-SAR)
            │   ├── Dataset (Optical-SAR)
            │   │   ├── train
            │   │   │   ├── opt_clear
            |   |   |   ├── opt_cloudy
            |   |   |   ├── SAR
            |   |   |   |   ├── VH
            |   |   |   |   ├── VV
            │   │   ├── test
            |   |   |   ├── opt_cloudy
            |   |   |   ├── SAR
            |   |   |   |   ├── VH
            |   |   |   |   ├── VV
            ├── groklst
            ├── tools

    """

    METAINFO = dict(dataset_type="OpticalSARDataset", task_name="cloud removal")

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
        img_suffix: str = '.png',
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
        self.img_suffix = img_suffix
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
        
        self.img = sorted(os.listdir(self.data_prefix["img"]))
        self.gt = sorted(os.listdir(self.data_prefix["gt"]))
        self.sar_vv = sorted(os.listdir(os.path.join(self.data_prefix["sar"], 'VV')))
        self.sar_vh = sorted(os.listdir(os.path.join(self.data_prefix["sar"], 'VH')))

        # assert (len(self.img) == len(self.gt)) and (len(self.sar_vv) == len(self.sar_vh)) and (len(self.img) == len(self.sar_vv)), f"The amount of optical image and SAR image data is inconsistent."
            
        data_list = []
        if self.use_ann_file:
            for idx in path_list:  # idx
                img = self.img[int(idx)]
                gt = self.gt[int(idx)]
                # sar_vv = self.sar_vv[int(idx)]
                sar_vh = self.sar_vh[int(idx)]

                basename, ext = osp.splitext(img)
                data = dict(key=basename)

                data["img_path"] = osp.join(self.data_prefix["img"], img)
                data["gt_path"] = osp.join(self.data_prefix["gt"], gt)
                # data["sar_vv_path"] = osp.join(self.data_prefix["sar"], 'VV', sar_vv)
                data["sar_vh_path"] = osp.join(self.data_prefix["sar"], 'VH', sar_vh)
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
            suffix=self.img_suffix,
            recursive=self.recursive,
        ):
            basename, ext = osp.splitext(img_path)
            if re.match(virtual_path, basename):
                img_path = img_path.replace(tmpl + ext, ext)
                path_list.append(img_path)

        return path_list
