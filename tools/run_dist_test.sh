#!/bin/bash

# sleep 8m

# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/sisr/srcnn/srcnn_x2_4xb1-10k_lst.py work_dirs/srcnn_x2_4xb1-10k_lst/train_20240920_053335_lst_z-score_L1loss/best_lst_RMSE_iter_10000.pth 4


# X2 guidance zscore 1 | lst minmax 2 ================================================================
# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/msg/msg_x2_4xb1-10k_lst.py work_dirs/msg_x2_4xb1-10k_lst/train_20240408_095941_gui_z-score_lst_min-max_L1loss/best_lst_RMSE_iter_10000.pth 4


# # export CUDA_VISIBLE_DEVICES=0,1,2,3 
# # PORT=29502 tools/dist_test.sh configs/gisr/svlrm/svlrm_x2_4xb1-10k_lst.py work_dirs/svlrm_x2_4xb1-10k_lst/train_20240408_132202_gui_z-score_lst_min-max_L1loss/best_lst_RMSE_iter_10000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/djfr/djfr_x2_4xb1-10k_lst.py work_dirs/djfr_x2_4xb1-10k_lst/train_20240408_133054_gui_z-score_lst_min-max_L1loss/best_lst_RMSE_iter_10000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/p2p/p2p_x2_4xb1-10k_lst.py work_dirs/p2p_x2_4xb1-10k_lst/train_20240408_133933_gui_z-score_lst_min-max_L1loss/best_lst_RMSE_iter_10000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/dsrn/dsrn_x2_4xb1-10k_lst.py work_dirs/dsrn_x2_4xb1-10k_lst/train_20240408_134741_gui_z-score_lst_min-max_L1loss/best_lst_RMSE_iter_10000.pth 4

# #
# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/fdsr/fdsr_x2_4xb1-10k_lst.py work_dirs/fdsr_x2_4xb1-10k_lst/train_20240408_140442_gui_z-score_lst_min-max_L1loss/best_lst_RMSE_iter_10000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/dkn/dkn_x2_4xb1-10k_lst.py work_dirs/dkn_x2_4xb1-10k_lst/train_20240408_141654_gui_z-score_lst_min-max_L1loss/best_lst_RMSE_iter_6000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/fdkn/fdkn_x2_4xb1-10k_lst.py work_dirs/fdkn_x2_4xb1-10k_lst/train_20240408_141813_gui_z-score_lst_min-max_L1loss/best_lst_RMSE_iter_10000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/ahmf/ahmf_x2_4xb1-10k_lst.py work_dirs/ahmf_x2_4xb1-10k_lst/train_20240408_103605_gui_z-score_lst_min-max_L1loss/best_lst_RMSE_iter_10000.pth 4

# # suft 
# # export CUDA_VISIBLE_DEVICES=0,1,2,3 
# # PORT=29502 tools/dist_test.sh configs/sisr/srcnn/srcnn_x2_4xb1-10k_lst.py work_dirs/srcnn_x2_4xb1-10k_lst/train_20240920_053335_lst_z-score_L1loss/best_lst_RMSE_iter_10000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/dagf/dagf_x2_4xb1-10k_lst.py work_dirs/dagf_x2_4xb1-10k_lst/train_20240408_143522_gui_z-score_lst_min-max_L1loss/best_lst_RMSE_iter_9000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/mocolsk/mocolsk_x2_4xb1-10k_lst.py work_dirs/mocolsk_x2_4xb1-10k_lst/train_20240626_110804_gui_z-score_lst_z-score_L1loss/best_lst_RMSE_iter_10000.pth 4



# ======================================================================================================================

# X2 guidance minmax 2 | lst minmax 2 ================================================================
# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/msg/msg_x2_4xb1-10k_lst.py work_dirs/msg_x2_4xb1-10k_lst/train_20240408_150313_gui_min-max_lst_min-max_L1loss/best_lst_RMSE_iter_10000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/svlrm/svlrm_x2_4xb1-10k_lst.py work_dirs/svlrm_x2_4xb1-10k_lst/train_20240409_011134_gui_min-max_lst_min-max_L1loss/best_lst_RMSE_iter_10000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/djfr/djfr_x2_4xb1-10k_lst.py work_dirs/djfr_x2_4xb1-10k_lst/train_20240408_151754_gui_min-max_lst_min-max_L1loss/best_lst_RMSE_iter_10000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/p2p/p2p_x2_4xb1-10k_lst.py work_dirs/p2p_x2_4xb1-10k_lst/train_20240409_010105_gui_min-max_lst_min-max_L1loss/best_lst_RMSE_iter_10000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/dsrn/dsrn_x2_4xb1-10k_lst.py work_dirs/dsrn_x2_4xb1-10k_lst/train_20240409_012525_gui_min-max_lst_min-max_L1loss/best_lst_RMSE_iter_9000.pth 4

# #
# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/fdsr/fdsr_x2_4xb1-10k_lst.py work_dirs/fdsr_x2_4xb1-10k_lst/train_20240409_015545_gui_min-max_lst_min-max_L1loss/best_lst_RMSE_iter_10000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/dkn/dkn_x2_4xb1-10k_lst.py work_dirs/dkn_x2_4xb1-10k_lst/train_20240409_023307_gui_min-max_lst_min-max_L1loss/best_lst_RMSE_iter_8000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/fdkn/fdkn_x2_4xb1-10k_lst.py work_dirs/fdkn_x2_4xb1-10k_lst/train_20240409_015603_gui_min-max_lst_min-max_L1loss/best_lst_RMSE_iter_10000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/ahmf/ahmf_x2_4xb1-10k_lst.py work_dirs/ahmf_x2_4xb1-10k_lst/train_20240408_155204_gui_min-max_lst_min-max_L1loss/best_lst_RMSE_iter_10000.pth 4

# # # suft 
# # # export CUDA_VISIBLE_DEVICES=0,1,2,3 
# # # PORT=29502 tools/dist_test.sh configs/sisr/srcnn/srcnn_x2_4xb1-10k_lst.py work_dirs/srcnn_x2_4xb1-10k_lst/train_20240920_053335_lst_z-score_L1loss/best_lst_RMSE_iter_10000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/dagf/dagf_x2_4xb1-10k_lst.py work_dirs/dagf_x2_4xb1-10k_lst/train_20240409_034550_gui_min-max_lst_min-max_L1loss/best_lst_RMSE_iter_9000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/mocolsk/mocolsk_x2_4xb1-10k_lst.py work_dirs/mocolsk_x2_4xb1-10k_lst/train_20240413_063647_gui_min-max_lst_min-max_L1loss/best_lst_RMSE_iter_10000.pth 4



# ======================================================================================================================

# X2 guidance minmax 2 | lst minmax 2 ================================================================
# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/msg/msg_x2_4xb1-10k_lst.py work_dirs/msg_x2_4xb1-10k_lst/train_20240920_055727_gui_min-max_lst_z-score_L1loss/best_lst_RMSE_iter_10000.pth 4


export CUDA_VISIBLE_DEVICES=0,1,2,3 
PORT=29502 tools/dist_test.sh configs/gisr/svlrm/svlrm_x2_4xb1-10k_lst.py work_dirs/svlrm_x2_4xb1-10k_lst/train_20240409_011134_gui_min-max_lst_min-max_L1loss/best_lst_RMSE_iter_10000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/djfr/djfr_x2_4xb1-10k_lst.py work_dirs/djfr_x2_4xb1-10k_lst/train_20240408_235254_gui_min-max_lst_z-score_L1loss/best_lst_RMSE_iter_10000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/p2p/p2p_x2_4xb1-10k_lst.py work_dirs/p2p_x2_4xb1-10k_lst/train_20240409_012315_gui_min-max_lst_z-score_L1loss/best_lst_RMSE_iter_10000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/dsrn/dsrn_x2_4xb1-10k_lst.py work_dirs/dsrn_x2_4xb1-10k_lst/train_20240409_012525_gui_min-max_lst_min-max_L1loss/best_lst_RMSE_iter_9000.pth 4

# #
# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/fdsr/fdsr_x2_4xb1-10k_lst.py work_dirs/fdsr_x2_4xb1-10k_lst/train_20240409_021402_gui_min-max_lst_z-score_L1loss/best_lst_RMSE_iter_10000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/dkn/dkn_x2_4xb1-10k_lst.py work_dirs/dkn_x2_4xb1-10k_lst/train_20240409_034512_gui_min-max_lst_z-score_L1loss/best_lst_RMSE_iter_9000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/fdkn/fdkn_x2_4xb1-10k_lst.py work_dirs/fdkn_x2_4xb1-10k_lst/train_20240409_021504_gui_min-max_lst_z-score_L1loss/best_lst_RMSE_iter_10000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/ahmf/ahmf_x2_4xb1-10k_lst.py work_dirs/ahmf_x2_4xb1-10k_lst/train_20240409_023532_gui_min-max_lst_z-score_L1loss/best_lst_RMSE_iter_10000.pth 4

# # suft 
# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/suft/suft_x2_4xb1-10k_lst.py work_dirs/suft_x2_4xb1-10k_lst/train_20240920_095205_gui_min-max_lst_min-max_L1loss/best_lst_RMSE_iter_10000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/dagf/dagf_x2_4xb1-10k_lst.py work_dirs/dagf_x2_4xb1-10k_lst/train_20240409_041509_gui_min-max_lst_z-score_L1loss/best_lst_RMSE_iter_10000.pth 4


# export CUDA_VISIBLE_DEVICES=0,1,2,3 
# PORT=29502 tools/dist_test.sh configs/gisr/mocolsk/mocolsk_x2_4xb1-10k_lst.py work_dirs/mocolsk_x2_4xb1-10k_lst/train_20240413_040543_gui_min-max_lst_z-score_L1loss/best_lst_RMSE_iter_10000.pth 4

