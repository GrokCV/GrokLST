# 🔥🔥 [GrokLST: Towards High-Resolution Benchmark and Toolkit for Land Surface Temperature Downscaling](https://arxiv.org/abs/2409.19835)  🔥🔥

Qun Dai, Chunyang Yuan, Yimian Dai, Yuxuan Li, Xiang Li, Kang Ni, Jianhui Xu, Xiangbo Shu, Jian Yang
----
Paper link: [GrokLST: Towards High-Resolution Benchmark and Toolkit for Land Surface Temperature Downscaling](https://arxiv.org/abs/2409.19835)


This repository is the official site for "GrokLST: Towards High-Resolution Benchmark and Toolkit for Land Surface Temperature Downscaling".

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



## Preface
🔥 The task of land surface temperature downscaling is similar to super-resolution tasks in the field of computer vision (CV). Researchers working on image super-resolution may want to take note of this! Currently, there are not many teams using deep learning methods for land surface temperature downscaling (most still rely on traditional methods), and there are almost no publicly available datasets for training deep learning models. This is considered a relatively underdeveloped field. However, land surface temperature downscaling has significant practical implications, especially in environmental science, climate research, agriculture, urban planning, and other areas. Therefore, this field holds great potential for development and real-world impact.

🔥 In light of this, we have proposed the first high-resolution land surface temperature downscaling dataset, GrokLST, along with the corresponding GrokLST toolkit. Our toolkit includes over 40 downscaling (or super-resolution) methods, covering both single-image super-resolution (SISR) and guided-image super-resolution (GISR) techniques. We hope our work can contribute to advancing the field!


## Introduction

**GrokLST DATASET DOWNLOAD at:**

* [Baidu Netdisk](https://pan.baidu.com/s/1-X2PHUBpFiq6JhtUWAUXLw?pwd=Grok)
* [OneDrive](https://1drv.ms/f/s!AmElF7K4aY9pgYEGx82XMMLm3n7zQQ?e=IsfN1I)

![GrokLST dataset](docs/groklst-dataset.png)

# MoCoLSK: Modality-Conditional Large Selective Kernel Module
![MoCoLSK module](docs/mocolsk.png)

# GrokLST Toolkit

  - [Abstract](#abstract)
  - [Preface](#preface)
  - [Introduction](#introduction)
  - [Installation](#installation)
    - [Step 1: Create a conda environment](#step-1-create-a-conda-environment)
    - [Step 2: Install PyTorch](#step-2-install-pytorch)
    - [Step 3: Install OpenMMLab 2.x Codebases](#step-3-install-openmmlab-2x-codebases)
    - [Step 4: Install `groklst`](#step-4-install-groklst)
  - [Folder Introduction](#folder-introduction)
    - [configs Folder](#configs-folder)
    - [data Folder and GrokLST Dataset Introduction](#data-folder-and-groklst-dataset-introduction)
    - [groklst Folder (Core Code)](#groklst-folder-core-code)
    - [tools Folder](#tools-folder)
  - [Training](#training)
    - [Single GPU Training](#single-gpu-training)
    - [Multi-GPU Training](#multi-gpu-training)
  - [Testing](#testing)
    - [Single GPU Testing](#single-gpu-testing)
    - [Multi-GPU Testing](#multi-gpu-testing)
  - [BUG LOG](#bug-log)
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


## Folder Introduction

- Overview of the GrokLST toolkit folder

```shell
GrokLST
    ├── configs  (Configuration files)
    ├── data (Datasets)
    ├── groklst (Core code)
    # ├── requirements (Environment dependencies)
    ├── tools (Training and testing tools)
    ├── work_dirs (Stores models, log files, etc.)
```


### configs Folder

```shell
# - Structure and function
configs
    ├── _base_
    │   ├── datasets (Configurations for different datasets, including pipelines, dataloaders, and evaluators for train, val, and test)
    │   │    ├── groklst_dataset
    │   │    │     ├──groklst_dataset_x2-256-512_sisr.py (SISR configuration, includes only LST without guidance data)
    │   │    │     ├──groklst_dataset_x2-256-512.py (GISR configuration, includes LST and guidance data)
    │   ├── schedules (Configurations for iterations, optimizers, etc.)
    │   ├── default_runtime.py (Default runtime configuration)
    ├── gisr (Guided Image Super-Resolution, GISR)
    ├── ...
    │   ├── mocolsk (Ours)
    │   │    ├── mocolsk_x2_4xb1-10k_groklst.py
    │   │    ├── mocolsk_x4_4xb1-10k_groklst.py
    │   │    ├── mocolsk_x8_4xb1-10k_groklst.py
    ├── ...
    ├── sisr (Single Image Super-Resolution, SISR)

```


### data Folder and GrokLST Dataset Introduction

1. Using the groklst dataset as an example to introduce its structure

```shell
    data
    ├──groklst (Each folder contains 641 mat files)
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

2. Using groklst / 30m as an example to explain the dataset:
- guidance folder: Each mat file contains 10 fields of data, namely "dem", "deepblue", "blue", "green", "red", "vre", "nir", "ndmvi", "ndvi", and "ndwi";

- lst folder: This is the LST (Land Surface Temperature) data, where the 30m resolution LST data can be used as the GT label, and the LST data of other resolutions (60m, 120m, 240m) can be regarded as low-resolution data that needs super-resolution;

- split folder: This is the dataset partitioning strategy, with train:val=6:1:3 and trainval=7:3. Note that in our paper [GrokLST: Towards High-Resolution Benchmark and
Toolkit for Land Surface Temperature Downscaling](https://arxiv.org/abs/2409.19835), we use trainval.txt as the training set, and the data indexed in test.txt as the validation and test sets.

3. The GrokLST dataset is obtained by cropping the original large image data with a 0.5 overlap rate, as shown in the table below:

| Resolution | scale | crop size | crop step | h/w | guidance size (h,w,c) |
| ---------- | ----- | --------- | --------- | --- | --------------------- |
| 30m        | -     | 512       | 256       | 512 | 512x512x10            |
| 60m        | x2    | 256       | 128       | 256 | 256x256x10            |
| 120m       | x4    | 128       | 64        | 128 | 128x128x10            |
| 240m       | x8    | 64        | 32        | 64  | 64x64x10              |


### groklst Folder (Core Code)

- Structure

```shell
groklst
    ├── datasets
    │   ├── transforms (Data processing pipelines)
    │   │    ├── dropping_bands.py  (Random band dropping)
    │   │    ├── formatting_data.py (Data packaging)
    │   │    ├── loading_data.py  (Data loading)
    │   │    ├── normalizing_data.py (Guidance data normalization)
    │   │    ├── padding_bands.py (Band padding)
    │   ├── groklst_dataset.py (GrokLSTDataset class)
    ├── evaluation
    │   │    ├── metrics  (Various evaluation metrics)
    ├── models
    │   ├── data_preprocessors
    │   │    ├── data_preprocessor.py  (Data preprocessor)
    │   ├── editors (All models are here!!!)
    │   │    ├── ...
    │   │    ├── mocolsk (Ours)
    │   │    ├── ...
    │   ├── losses
    │   │    ├── loss_wrapper.py
    │   │    ├── pixelwise_loss.py (Custom SmoothL1Loss)
    ├── visualization
    │   ├── custom_concat_visualizer.py  (Custom visualizer)
    │   ├── vis_backend.py
    │   ├── visualizer.py
```


### tools Folder

- Structure as follows

```shell
tools  (Entry points for training and testing!!!)
    ├── dist_test.sh (Distributed testing script)
    ├── dist_train.sh (Distributed training script)
    ├── test.py (Single GPU testing script)
    ├── train.py (Single GPU training script)
```


## Training

### Single GPU Training

- Single GPU training command (default using GPU0)：

```shell
python tools/train.py ${CONFIG_FILE}

# Example with MoCoLSK-Net
python tools/train.py configs/gisr/mocolsk/mocolsk_x8_4xb1-10k_groklst.py
```


### Multi-GPU Training

- Multi-GPU training command (assuming a 4-GPU machine):

```shell
# export CUDA_VISIBLE_DEVICES=0,1,2,3
PORT=29500 tools/dist_train.sh ${CONFIG_FILE} 4

# Example with MoCoLSK-Net
PORT=29500 tools/dist_train.sh configs/gisr/mocolsk/mocolsk_x8_4xb1-10k_groklst.py 4
```


## Testing

### Single GPU Testing

- Single GPU testing command (default using GPU0):

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}

# Example with MoCoLSK-Net
python tools/test.py configs/gisr/mocolsk/mocolsk_x8_4xb1-10k_groklst.py your/model/path.pth
```


### Multi-GPU Testing

- Multi-GPU testing command (assuming a 4-GPU machine):

```shell
# export CUDA_VISIBLE_DEVICES=0,1,2,3
tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} 4

# Example with MoCoLSK-Net
PORT=29500 tools/dist_test.sh configs/gisr/mocolsk/mocolsk_x8_4xb1-10k_groklst.py your/model/path.pth 4
```


<!-- ## Model Zoo and Benchmark
>>>>>>> a9207f24ff19bb19abdbe01e7b32a686cee15e49

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
            <a href=""> BaiduDisk </a> | <a href=""> OneDirve </a>
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
            <a href=""> BaiduDisk </a> | <a href=""> OneDirve </a>
        </td>
    <tr>
</table> -->


## BUG LOG
When running dist_test.sh or test.py, you may encounter the following issue:

```shell
"The model and loaded state dict do not match exactly

unexpected key in source state_dict: generator.module.conv1.weight, generator.module.conv1.bias, generator.module.conv2.weight, generator.module.conv2.bias, generator.module.conv3.weight, generator.module.conv3.bias

missing keys in source state_dict: generator.conv1.weight, generator.conv1.bias, generator.conv2.weight, generator.conv2.bias, generator.conv3.weight, generator.conv3.bias"
```

Problem Analysis:
- Initially, you might suspect that the network model and the state_dict saved in the checkpoint do not match, but this is not the case.
- The issue is mainly caused by the parameter revise_keys=[(r'^module//.', '')] in the function _load_checkpoint_to_model (around line 585) in mmengine.runner.checkpoint.py.
- '^module//.' is a regular expression pattern that aims to replace keys like "generator.module.conv1.weight" with "generator.conv1.weight", effectively removing "module." from "generator.module.conv1.weight".
- However, since "generator.module.conv1.weight" does not begin with "module.", it doesn't match the pattern '^module//.'.

Solution:
- In the mmengine.runner.runner.py, within the load_checkpoint function of the Runner class (around line 2111), replace the parameter revise_keys=[(r'^module//.', '')] with revise_keys=[(r'/bmodule.', '')]. This change ensures that keys like "generator.module.conv1.weight" will correctly be replaced with "generator.conv1.weight", effectively removing "module.".

This solution should help resolve the key mismatch when loading checkpoints.
<!-- ## Get Started

Please see [get_started.md](https://github.com/open-mmlab/mmagic/blob/main/docs/en/get_started/overview.md) for the basic usage of GrokLST toolkit. -->


## Acknowledgement

<!-- Our GrokLST toolkit is based on [MMAgic](https://mmagic.readthedocs.io/en/latest/) developed by [OpenMMLAB](https://openmmlab.com/), and thanks to mmagic community! -->
The authors would like to thank the International Research Center of Big Data for Sustainable Development Goals (CBAS) for kindly providing the SDGSAT-1 data.

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

