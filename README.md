# 🔥🔥 [GrokLST: Towards High-Resolution Benchmark and Toolkit for Land Surface Temperature Downscaling](https://arxiv.org/abs/2409.19835)  🔥🔥

Qun Dai, Chunyang Yuan, Yimian Dai, Yuxuan Li, Xiang Li, Kang Ni, Jianhui Xu, Xiangbo Shu, Jian Yang
----
Paper link: [GrokLST: Towards High-Resolution Benchmark and Toolkit for Land Surface Temperature Downscaling](https://arxiv.org/abs/2409.19835)


This repository is the official site for "GrokLST: Towards High-Resolution Benchmark and Toolkit for Land Surface Temperature Downscaling".

🔥 地表温度降尺度任务和 CV 领域的超分任务类似，做图像超分任务的各位大佬可以关注一下！目前用深度学习方法做地表温度降尺度任务的团队并不多（大多都是采用传统方法），基本没有用于深度学习模型训练用的公开数据集，算得上是一个相对滞后的领域。但地表温度降尺度任务具有重要的现实意义，特别是在环境科学、气候研究、农业、城市规划等多个领域。因此，地表温度降尺度领域有着非常好的发展前景和现实意义。

🔥 鉴于以上现状，我们提出了第一个高分辨率地表温度降尺度数据集 GrokLST 以及相应工具箱 GrokLST toolkit，我们的工具箱包含了 40+ 降尺度（即超分）方法，包括单张图像降尺度方法（SISR）以及引导图像降尺度方法（GISR）。希望我们的工作能够促使该领域更好地向前发展！
## Abstract

Land Surface Temperature (LST) is a critical parameter for environmental studies, but obtaining high-resolution LST data remains challenging due to the spatio-temporal trade-off in satellite remote sensing.
Guided LST downscaling has emerged as a solution, but current methods often neglect spatial non-stationarity and lack a open-source ecosystem for deep learning methods.
To address these limitations, we propose the Modality-Conditional Large Selective Kernel (MoCoLSK) Networks, a novel architecture that dynamically fuses multi-modal data through modality-conditioned projections. MoCoLSK re-engineers our previous LSKNet to achieve a confluence of dynamic receptive field adjustment and multi-modal feature integration, leading to enhanced LST prediction accuracy.
Furthermore, we establish the GrokLST project, a comprehensive open-source ecosystem featuring the GrokLST dataset, a high-resolution benchmark, and the GrokLST toolkit, an open-source PyTorch-based toolkit encapsulating MoCoLSK alongside 40+ state-of-the-art approaches.
Extensive experimental results validate MoCoLSK's effectiveness in capturing complex dependencies and subtle variations within multispectral data, outperforming existing methods in LST downscaling.
Our code, dataset, and toolkit are available at https://github.com/GrokCV/GrokLST.

<!-- # todo 放个数据集图像 -->
<!-- ![GrokLST dataset](docs/groklst-region.png) -->
![MoCoLSK-Net](docs/mocolsk-net.png)


## Introduction

**DATASET DOWNLOAD at:**


* [Baidu Netdisk](https://pan.baidu.com/s/1-X2PHUBpFiq6JhtUWAUXLw?pwd=Grok)
* [OneDrive](https://1drv.ms/f/s!AmElF7K4aY9pgYEGx82XMMLm3n7zQQ?e=IsfN1I)

![GrokLST dataset](docs/groklst-dataset.png)

# MoCoLSK: Modality-Conditional Large Selective Kernel Module
![MoCoLSK module](docs/mocolsk.png)

# GrokLST Toolkit

- [🔥🔥 GrokLST: Towards High-Resolution Benchmark and Toolkit for Land Surface Temperature Downscaling  🔥🔥](#-groklst-towards-high-resolution-benchmark-and-toolkit-for-land-surface-temperature-downscaling--)
  - [Qun Dai, Chunyang Yuan, Yimian Dai, Yuxuan Li, Xiang Li, Kang Ni, Jianhui Xu, Xiangbo Shu, Jian Yang](#qun-dai-chunyang-yuan-yimian-dai-yuxuan-li-xiang-li-kang-ni-jianhui-xu-xiangbo-shu-jian-yang)
  - [Abstract](#abstract)
  - [Introduction](#introduction)
- [MoCoLSK: Modality-Conditional Large Selective Kernel Module](#mocolsk-modality-conditional-large-selective-kernel-module)
- [GrokLST Toolkit](#groklst-toolkit)
  - [Installation](#installation)
    - [Step 1: Create a conda environment](#step-1-create-a-conda-environment)
    - [Step 2: Install PyTorch](#step-2-install-pytorch)
    - [Step 3: Install OpenMMLab 2.x Codebases](#step-3-install-openmmlab-2x-codebases)
    - [Step 4: Install `groklst`](#step-4-install-groklst)
  - [⭐文件夹介绍](#文件夹介绍)
    - [configs 文件夹](#configs-文件夹)
    - [data 文件夹以及 GrokLST 数据集简介](#data-文件夹以及-groklst-数据集简介)
    - [groklst 文件夹 (核心代码)](#groklst-文件夹-核心代码)
    - [tools 文件夹](#tools-文件夹)
  - [🚀训练](#训练)
    - [单卡训练](#单卡训练)
    - [多卡训练](#多卡训练)
  - [🚀测试](#测试)
    - [单卡测试](#单卡测试)
    - [多卡测试](#多卡测试)
  - [Model Zoo and Benchmark](#model-zoo-and-benchmark)
    - [Leaderboard](#leaderboard)
    - [Model Zoo](#model-zoo)
      - [Method A](#method-a)
      - [Method B](#method-b)
  - [Get Started](#get-started)
  - [Acknowledgement](#acknowledgement)
  - [Citation](#citation)
  - [License](#license)


## Installation

### Step 1: Create a conda environment

```shell
conda create --name groklst python=3.9 -y
source activate groklst
```

### Step 2: Install PyTorch

```shell
# CUDA 12.1
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### Step 3: Install OpenMMLab 2.x Codebases

```shell
# openmmlab codebases
pip install -U openmim dadaptation chardet --no-input
# mim install mmengine "mmcv==2.1.0" "mmdet>=3.0.0" "mmpretrain>=1.0.0rc7"
mim install mmengine "mmcv==2.1.0" "mmdet>=3.0.0" "mmsegmentation>=1.0.0" "mmpretrain>=1.0.0rc7" 'mmagic'
# other dependencies
pip install -U ninja scikit-image --no-input
pip install kornia==0.6.5 # 0.7.1
pip install albumentations==1.3.1
pip install diffusers==0.24.0
```

### Step 4: Install `groklst`

```shell
python setup.py develop
```

**Note**: make sure you have `cd` to the root directory of `GrokLST`

```shell
git clone git@github.com:GrokCV/GrokLST.git
cd GrokLST
```


## ⭐文件夹介绍

- GrokLST toolkit 整体文件夹介绍

```shell
GrokLST
    ├── configs  (配置文件夹)
    ├── data (数据集)
    ├── groklst (核心代码)
    # ├── requirements (环境依赖)
    ├── tools (训练测试工具)
    ├── work_dirs (存放模型及日志文件等)
```


### configs 文件夹

```shell
# - 结构及作用
configs
    ├── _base_
    │   ├── datasets (不同数据集的配置，包括 train、val、test 的 pipeline、dataloader 和 evaluator 等)
    │   │    ├── groklst_dataset
    │   │    │     ├──groklst_dataset_x2-256-512_sisr.py (SISR 配置，不包括引导数据，只含有 LST)
    │   │    │     ├──groklst_dataset_x2-256-512.py (GISR 配置，包括 LST 和引导数据)
    │   ├── schedules (迭代次数及优化器等配置)
    │   ├── default_runtime.py (默认运行时配置)
    ├── gisr (Guided Image Super-Resolution, GISR)
    ├── ...
    │   ├── mocolsk (Ours)
    │   │    ├── mocolsk_x2_4xb1-10k_groklst.py
    │   │    ├── mocolsk_x4_4xb1-10k_groklst.py
    │   │    ├── mocolsk_x8_4xb1-10k_groklst.py
    ├── ...
    ├── sisr (Single Image Super-Resolution, SISR)
```


### data 文件夹以及 GrokLST 数据集简介

1. 以 groklst 数据集为例来介绍其存放结构

```shell
  data
    ├──groklst (每个文件夹下有 641 个 mat 文件)
        ├── 30m
        │   ├── guidance (HR guidance)
        │   ├── lst (LR LST)
        ├── 60m
        │   ├── guidance
        │   ├── lst
        ├── 120m
        │   ├── guidance
        │   ├── lst
        ├── 240m
        │   ├── guidance
        │   ├── lst
        ├── split (train:val:test=6:1:3)
        │   ├── train.txt
        │   ├── val.txt
        │   ├── trainval.txt (tarin.txt + val.txt)
        │   ├── test.txt
```

2. 以 groklst / 30m 为例, 来说明该数据集：
- guidance 文件夹：其中每个 mat 文件中包含 10 个字段数据，分别是 "dem", "deepblue", "blue", "green", "red", "vre", "nir", "ndmvi", "ndvi", "ndwi";

- lst 文件夹：是 LST（Land Surface Temperature）数据，其中 30m 分辨率的 LST 数据可当做标签 GT，其他分辨率的 LST 数据（60m, 120m, 240m）都可视为需要超分的低分辨率数据；

- split 文件夹：是数据集的划分策略，train:val:test=6:1:3，trainval:test=7:3, 注意我们的论文 [GrokLST: Towards High-Resolution Benchmark and
Toolkit for Land Surface Temperature Downscaling](https://arxiv.org/abs/2409.19835) 中采用 trainval.txt 当做训练集，test.txt 中指定下标的数据当验证集和测试集）;

3. GrokLST 数据集是对原始大影像数据采用 0.5 的重叠率进行裁剪获取的，详见下表：

| 分辨率 | scale | crop size | crop step | h/w | guidance size (h,w,c) |
| ------ | ----- | --------- | --------- | --- | --------------------- |
| 30m    | -     | 512       | 256       | 512 | 512x512x10            |
| 60m    | x2    | 256       | 128       | 256 | 256x256x10            |
| 120m   | x4    | 128       | 64        | 128 | 128x128x10            |
| 240m   | x8    | 64        | 32        | 64  | 64x64x10              |


### groklst 文件夹 (核心代码)

- 结构

```shell
groklst
    ├── datasets
    │   ├── transforms (数据处理的 pipeline)
    │   │    ├── dropping_bands.py  (随机丢弃波段)
    │   │    ├── formatting_data.py (打包数据)
    │   │    ├── loading_data.py  (加载数据)
    │   │    ├── normalizing_data.py (归一化引导数据)
    │   │    ├── padding_bands.py (填充波段)
    │   ├── groklst_dataset.py (GrokLSTDataset 类)
    ├── evaluation
    │   │    ├── metrics  (各种衡量指标)
    ├── models
    │   ├── data_preprocessors
    │   │    ├── data_preprocessor.py  (数据预处理器)
    │   ├── editors (所有模型在此！！！)
    │   │    ├── ...
    │   │    ├── mocolsk (Ours)
    │   │    ├── ...
    │   ├── losses
    │   │    ├── loss_wrapper.py
    │   │    ├── pixelwise_loss.py (自定义的 SmoothL1Loss)
    ├── visualization
    │   ├── custom_concat_visualizer.py  (自定义的可视化器)
    │   ├── vis_backend.py
    │   ├── visualizer.py
```


### tools 文件夹

- 结构如下

```shell
tools  (训练测试入口！！！)
    ├── dist_test.sh (分布式测试脚本)
    ├── dist_train.sh (分布式训练脚本)
    ├── test.py (单卡测试脚本)
    ├── train.py (单卡训练脚本)
```


## 🚀训练

### 单卡训练

- 单卡训练命令 (默认使用 GPU0)：

```shell
python tools/train.py ${CONFIG_FILE}

# 以 MoCoLSK-Net 为例
python tools/train.py configs/gisr/mocolsk/mocolsk_x8_4xb1-10k_groklst.py
```


### 多卡训练

- 多卡训练命令 (假设是四卡机)：

```shell
# export CUDA_VISIBLE_DEVICES=0,1,2,3
PORT=29500 tools/dist_train.sh ${CONFIG_FILE} 4

# 以 MoCoLSK-Net 为例
PORT=29500 tools/dist_train.sh configs/gisr/mocolsk/mocolsk_x8_4xb1-10k_groklst.py 4
```


## 🚀测试

### 单卡测试

- 单卡测试命令 (默认使用 GPU0)：

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}

# 以 MoCoLSK-Net 为例：
python tools/test.py configs/gisr/mocolsk/mocolsk_x8_4xb1-10k_groklst.py your/model/path.pth
```


### 多卡测试

- 多卡测试命令 (假设是四卡机)：

```shell
# export CUDA_VISIBLE_DEVICES=0,1,2,3
tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} 4

# 以 MoCoLSK-Net 为例：
PORT=29500 tools/dist_test.sh configs/gisr/mocolsk/mocolsk_x8_4xb1-10k_groklst.py your/model/path.pth 4
```


## Model Zoo and Benchmark

**Note: Both passwords for BaiduYun and OneDrive is `grok`**.

### Leaderboard

### Model Zoo

#### Method A

<table>
    <tr>
        <td>Model</td>
        <td>mAP</td>
        <td>#Params</td>
        <td>FLOPs</td>
        <td>Config</td>
        <td>Training Log</td>
        <td>Checkpoint</td>
        <td>Visualization</td>
    <tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td colspan="4">
            <a href=""> 百度网盘 </a> | <a href=""> OneDirve </a>
        </td>
    <tr>
</table>

#### Method B

<table>
    <tr>
        <td>Model</td>
        <td>mAP</td>
        <td>#Params</td>
        <td>FLOPs</td>
        <td>Config</td>
        <td>Training Log</td>
        <td>Checkpoint</td>
        <td>Visualization</td>
    <tr>
    <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td colspan="4">
            <a href=""> 百度网盘 </a> | <a href=""> OneDirve </a>
        </td>
    <tr>
</table>


## Get Started

Please see [get_started.md](https://github.com/open-mmlab/mmagic/blob/main/docs/en/get_started/overview.md) for the basic usage of GrokLST toolkit. More user guides see [User Guides](https://mmagic.readthedocs.io/en/latest/)!


## Acknowledgement

Our GrokLST toolkit is based on [MMAgic](https://mmagic.readthedocs.io/en/latest/) developed by [OpenMMLAB](https://openmmlab.com/), and thanks to mmagic community!

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```bibtex
@article{dai2024GrokLST,
	title={GrokLST: Towards High-Resolution Benchmark and Toolkit for Land Surface Temperature Downscaling},
	author={Dai, Qun and Yuan, Chunyang and Dai, Yimian and Li, Yuxuan and Li, Xiang and Ni, Kang and Xu, Jianhui and Shu, Xiangbo and Yang, Jian},
	year={2024},
	journal={arXiv},
}
```

## License

This project is released under the [Attribution-NonCommercial 4.0 International](LICENSE).

