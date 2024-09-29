#!/bin/bash

# 进入工作目录
cd /data1/dym/GrokLST-Dev/

# 设置CUDA可见的设备
export CUDA_VISIBLE_DEVICES=1,2,3,4

# 使用conda环境中的Python解释器来运行Python脚本
source /data1/dym/anaconda3/bin/activate groklst

# 启动训练脚本
python ./tools/dist_train.sh ./configs/fdsr/fdsr_x2_4xb1-10k_lst.py 4

# 退出conda环境
# conda deactivate

