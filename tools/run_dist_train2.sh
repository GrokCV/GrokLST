#!/bin/bash

# sleep 8m

# export CUDA_VISIBLE_DEVICES=2,3


# CUDA_VISIABLE_DEVICES=0,1,2,3 PORT=29502 tools/dist_train.sh configs/gisr/mocolsk_abs/mocolsk_x8_4xb1-10k_lst_MoCoLSKBlock.py 2

# CUDA_VISIABLE_DEVICES=0,1,2,3 PORT=29502 tools/dist_train.sh configs/gisr/mocolsk_abs/mocolsk_x8_4xb1-10k_lst_MoCoLSKBlock_SCSC.py 2

# stages
# CUDA_VISIABLE_DEVICES=0,1,2,3 PORT=29502 tools/dist_train.sh configs/gisr/mocolsk_stages/mocolsk_x8_4xb1-10k_lst_MoCoLSKBlock_s1.py 2

# CUDA_VISIABLE_DEVICES=0,1,2,3 PORT=29502 tools/dist_train.sh configs/gisr/mocolsk_stages/mocolsk_x8_4xb1-10k_lst_MoCoLSKBlock_s2.py 2


# CUDA_VISIABLE_DEVICES=0,1,2,3 PORT=29502 tools/dist_train.sh configs/gisr/mocolsk_stages/mocolsk_x8_4xb1-10k_lst_MoCoLSKBlock_s3.py 2

# CUDA_VISIABLE_DEVICES=0,1,2,3 PORT=29502 tools/dist_train.sh configs/gisr/mocolsk_stages/mocolsk_x8_4xb1-10k_lst_MoCoLSKBlock_s5.py 2

# dmlp type!
# CUDA_VISIABLE_DEVICES=0,1,2,3 PORT=29502 tools/dist_train.sh configs/gisr/mocolsk_dmlp/mocolsk_x8_4xb1-10k_lst_MoCoLSKBlock_b1_k3.py 2

# CUDA_VISIABLE_DEVICES=0,1,2,3 PORT=29502 tools/dist_train.sh configs/gisr/mocolsk_dmlp/mocolsk_x8_4xb1-10k_lst_MoCoLSKBlock_c1_k3.py 2

# dmlp layers A
# CUDA_VISIABLE_DEVICES=0,1,2,3 PORT=29502 tools/dist_train.sh configs/gisr/mocolsk_dmlp/mocolsk_x8_4xb1-10k_lst_MoCoLSKBlock_a2_k3.py 2

# CUDA_VISIABLE_DEVICES=0,1,2,3 PORT=29502 tools/dist_train.sh configs/gisr/mocolsk_dmlp/mocolsk_x8_4xb1-10k_lst_MoCoLSKBlock_a3_k3.py 2

# dmlp layers B
# CUDA_VISIABLE_DEVICES=0,1,2,3 PORT=29502 tools/dist_train.sh configs/gisr/mocolsk_dmlp/mocolsk_x8_4xb1-10k_lst_MoCoLSKBlock_b2_k3.py 2

# CUDA_VISIABLE_DEVICES=0,1,2,3 PORT=29502 tools/dist_train.sh configs/gisr/mocolsk_dmlp/mocolsk_x8_4xb1-10k_lst_MoCoLSKBlock_b3_k3.py 2

# # dmlp layers C
# CUDA_VISIABLE_DEVICES=0,1,2,3 PORT=29502 tools/dist_train.sh configs/gisr/mocolsk_dmlp/mocolsk_x8_4xb1-10k_lst_MoCoLSKBlock_c2_k3.py 2

# CUDA_VISIABLE_DEVICES=0,1,2,3 PORT=29502 tools/dist_train.sh configs/gisr/mocolsk_dmlp/mocolsk_x8_4xb1-10k_lst_MoCoLSKBlock_c3_k3.py 2




# CUDA_VISIABLE_DEVICES=0,1,2,3 PORT=29502 tools/dist_train.sh configs/gisr/mocolsk_dmlp/mocolsk_x8_4xb1-10k_lst_MoCoLSKBlock_b4_k3.py 2

# CUDA_VISIABLE_DEVICES=0,1,2,3 PORT=29502 tools/dist_train.sh configs/gisr/mocolsk_dmlp/mocolsk_x8_4xb1-10k_lst_MoCoLSKBlock_c4_k3.py 2

# CUDA_VISIABLE_DEVICES=0,1,2,3 PORT=29502 tools/dist_train.sh configs/gisr/mocolsk_dmlp/mocolsk_x8_4xb1-10k_lst_MoCoLSKBlock_a4_k3.py 2

# CUDA_VISIABLE_DEVICES=0,1,2,3 PORT=29502 tools/dist_train.sh configs/gisr/mocolsk_dmlp/mocolsk_x8_4xb1-10k_lst_MoCoLSKBlock_c1_k11.py 2


# CUDA_VISIABLE_DEVICES=0,1,2,3 PORT=29502 tools/dist_train.sh configs/gisr/mocolsk_abs/mocolsk_x8_4xb1-10k_lst_MoCoOneLKBlock.py 2


# CUDA_VISIABLE_DEVICES=0,1,2,3 PORT=29502 tools/dist_train.sh configs/gisr/mocolsk_abs/mocolsk_x8_4xb1-10k_lst_MoCoLSKBlock_SCSS.py 2


# CUDA_VISIABLE_DEVICES=0,1,2,3 PORT=29502 tools/dist_train.sh configs/gisr/mocolsk_abs/mocolsk_x8_4xb1-10k_lst_MoCoOneLKBlock.py 2


# out of memary
# CUDA_VISIABLE_DEVICES=0,1,2,3 PORT=29502 tools/dist_train.sh configs/gisr/mocolsk_dim/mocolsk_x8_4xb1-10k_lst_MoCoLSKBlock_d48.py 2



# CUDA_VISIABLE_DEVICES=0,1,2,3 PORT=29502 tools/dist_train.sh configs/gisr/mocolsk_abs/mocolsk_x8_4xb1-10k_lst_MoCoLSKWithSmallKernelBlock_k7d1p3_k9d4p16.py 2

sleep 55m


export CUDA_VISIBLE_DEVICES=0,1,2,3
PORT=29500 tools/dist_train.sh configs/gisr/mocolsk/mocolsk_x2_4xb1-10k_lst.py 4





