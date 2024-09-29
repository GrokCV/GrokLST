# Copyright (c) OpenMMLab. All rights reserved.
import logging
import re
from typing import Sequence
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from mmengine.visualization import Visualizer
from scipy import io as sio
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mmagic.registry import VISUALIZERS
from mmagic.structures import DataSample
from mmagic.utils import print_colored_log


@VISUALIZERS.register_module()
class ConcatLSTVisualizer(Visualizer):
    """Visualize multiple LSTs by concatenation.

    This visualizer will horizontally concatenate images belongs to different
    keys and vertically concatenate images belongs to different frames to
    visualize.

    Image to be visualized can be:
        - torch.Tensor or np.array
        - Image sequences of shape (T, C, H, W)
        - Multi-channel image of shape (1/3, H, W)
        - Single-channel image of shape (C, H, W)

    Args:
        fn_key (str): key used to determine file name for saving image.
            Usually it is the path of some input image. If the value is
            `dir/basename.ext`, the name used for saving will be basename.
        img_keys (str): keys, values of which are images to visualize.
        pixel_range (dict): min and max pixel value used to denormalize images,
            note that only float array or tensor will be denormalized,
            uint8 arrays are assumed to be unnormalized.
        bgr2rgb (bool): whether to convert the image from BGR to RGB.
        name (str): name of visualizer. Default: 'visualizer'.
        *args and \**kwargs: Other arguments are passed to `Visualizer`. # noqa
    """
    MAPPING = {"input": "LR_LST", "pred_img": "Pred_LST", "gt_img": "GT_LST"}

    def __init__(
        self,
        fn_key: str,  # filename key for saving img.
        img_keys: Sequence[str],
        mask_keys: Sequence[str] = None,
        apply_mask: bool = True,
        name: str = "visualizer",
        use_color_map: bool = False, # 不使用 color_map 意味着保存 pred_lst 数据为 mat 格式
        show_imgs: bool = False, # 是否显示图像
        add_title: bool = False, # 图片上是否添加标题
        show_diff: bool = False, # 是否在最右侧添加 pred_img - gt_img
        show_color_bar: bool = False, # 是否在最右侧显示色条，但是添加色条之后的图像会变小一点
        *args,
        **kwargs,
    ) -> None:
        super().__init__(name, *args, **kwargs)
        self.fn_key = fn_key
        self.img_keys = img_keys
        self.apply_mask = apply_mask
        self.use_color_map = use_color_map
        self.mask_keys = mask_keys
        self.show_imgs = show_imgs
        self.add_title = add_title
        self.show_diff = show_diff
        self.show_color_bar = show_color_bar

        if self.mask_keys is not None:
            assert len(self.img_keys) == len(self.mask_keys)

    def add_datasample(self, data_sample: DataSample, step=0) -> None:
        """Concatenate image and draw.

        Args:
            input (torch.Tensor): Single input tensor from data_batch.
            data_sample (DataSample): Single data_sample from data_batch.
            output (DataSample): Single prediction output by model.
            step (int): Global step value to record. Default: 0.
        """
        # Note:
        # with LocalVisBackend and default arguments, we have:
        # self.save_dir == runner._log_dir / 'vis_data'

        merged_dict = {
            **data_sample.to_dict(),
        }

        if "output" in merged_dict.keys():
            merged_dict.update(**merged_dict["output"])

        fn = merged_dict[self.fn_key]
        if isinstance(fn, list):
            fn = fn[0]
        fn = re.split(r" |/|\\", fn)[-1]
        fn = fn.split(".")[0]

        img_list = []
        for k in self.img_keys:
            if k not in merged_dict:
                print_colored_log(f'Key "{k}" not in data_sample or outputs', level=logging.WARN)
                continue

            img = merged_dict[k]

            # PixelData
            if isinstance(img, dict) and ("data" in img):
                img = img["data"]

            # Tensor to array: chw->hwc
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
                if img.ndim == 3:
                    img = img.transpose(1, 2, 0)
                elif img.ndim == 4:
                    img = img.transpose(0, 2, 3, 1)

            # concat frame vertically
            if img.ndim == 4:
                img = np.concatenate(img, axis=0)

            img_list.append(img)

        # visualize and save LST images list.
        if self.use_color_map:
            self.vis_save_lst(merged_dict, img_list, fn)
        else:
            self.save_to_mat(img_list, fn)
    
    
    def save_to_mat(self, img_list: Sequence[np.array], filename: str):
        field_names = []
        for i, img_key in enumerate(self.img_keys):
            field_name = str(self.MAPPING[img_key])
            field_names.append(field_name)

        # 构建一个字典，将 field_names 中的元素与 img_list 中的元素一一对应
        mdict = {field_name: img for field_name, img in zip(field_names, img_list)}

        filename = filename + ".mat"
        save_path = os.path.join(self._vis_backends["LocalVisBackend"]._save_dir, filename)
        sio.savemat(save_path, mdict=mdict)

    def vis_save_lst(self, merged_dict: dict, img_list: Sequence[np.array], filename: str):
        
        if self.show_diff:
            num_imgs = len(img_list) + 1
        else:
            num_imgs = len(img_list)
        # 创建图和坐标轴
        fig, axes = plt.subplots(1, num_imgs, figsize=(5 * num_imgs, 5))
        # 创建一个颜色映射，将0显示为白色，其他非零值显示为不同颜色
        # "jet" "hot" "plasma" "magma" "inferno" "cividis" "viridis"
        cmap = plt.get_cmap("viridis")
        if num_imgs == 1:  # 单个图像的情况
            axes = [axes]

        for i, data in enumerate(img_list):
            img_key = self.img_keys[i]
            if self.mask_keys is not None:
                mask_key = self.mask_keys[i]
                mask = merged_dict[mask_key].detach().cpu().numpy()
                mask = mask == 1
                mask = np.transpose(mask, (1, 2, 0))
                # we set nan value for data[~mask], we need white background.
                data[~mask] = np.nan

                # 设置颜色映射的范围，使得0对应的颜色是白色
                vmin = np.min(data[mask])
                vmax = np.max(data[mask])
            else:
                vmin = np.min(data)
                vmax = np.max(data)
            ax = axes[i]
            img = axes[i].imshow(data, cmap=cmap, interpolation=None, vmin=vmin, vmax=vmax)
            if self.add_title:
                ax.set_title(str(self.MAPPING[img_key]))

            # 打开坐标轴边框
            ax.set_frame_on(True)

            # 隐藏刻度
            ax.set_xticks([])
            ax.set_yticks([])
            # 隐藏坐标轴值
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        if self.show_diff:
            pred_img = img_list[1]
            gt_img = img_list[2]
            diff_data = pred_img - gt_img
            if self.mask_keys is not None:
                mask_key = "hr_mask"
                mask = merged_dict[mask_key].detach().cpu().numpy()
                mask = mask == 1
                mask = np.transpose(mask, (1, 2, 0))
                # we set nan value for diff_data[~mask], we need white background.
                diff_data[~mask] = np.nan

                # 设置颜色映射的范围，使得0对应的颜色是白色
                vmin = np.min(diff_data[mask])
                vmax = np.max(diff_data[mask])
            else: 
                vmin = np.min(diff_data)
                vmax = np.max(diff_data)
            # print(f"vmin={vmin}, vmax={vmax}")
            ax = axes[-1]
            ax.set_title("Pred-GT")
            diff_img = axes[-1].imshow(diff_data, cmap=cmap, interpolation=None, vmin=vmin, vmax=vmax)
            # ax.set_title("GT-Pred")
            # diff_img = axes[-1].imshow(-diff_data, cmap=cmap, interpolation=None, vmin=vmin, vmax=vmax)

            # 打开坐标轴边框
            ax.set_frame_on(True)

            # 隐藏刻度
            ax.set_xticks([])
            ax.set_yticks([])
            # 隐藏坐标轴值
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            # add cbar for diff
            if self.show_color_bar:
                divider = make_axes_locatable(axes[-1])
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = plt.colorbar(diff_img, cax=cax, orientation="vertical")
                cbar.set_label("Temperature/K")

        # # 在最右侧子图的右侧添加颜色条
        if self.show_color_bar:
            divider = make_axes_locatable(axes[-2])
            # pad=0.1  颜色条与其所附着的轴之间的距离是轴宽度的 10%
            cax = divider.append_axes("right", size="5%", pad=0.1) 
            cbar = plt.colorbar(img, cax=cax, orientation="vertical")
            cbar.set_label("Temperature/K")

        # add title
        if self.add_title:
            plt.suptitle(filename)
        # 保存为 png 格式的图片
        filename = filename + ".png"
        save_path = os.path.join(self._vis_backends["LocalVisBackend"]._save_dir, filename)
        plt.savefig(save_path, bbox_inches="tight", dpi=300, transparent=False, pad_inches=0)

        # 显示图形
        if self.show_imgs:
            plt.show()
            plt.pause(1)  # 显示图像 1 秒钟
        plt.close()
