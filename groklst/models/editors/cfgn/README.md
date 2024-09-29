# CFGN-PyTorch

This repository is an official PyTorch implementation of CFGN. Our code is based on [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch). If you want to learn more about our code, please refer to [EDSR-PyTorch](https://github.com/sanghyun-son/EDSR-PyTorch).

## Train (Demo)
If you want to train CFGN, please ``cd */CFGN-PyTorch/src`` and run the following command in bash.

Train CFGN with upsampling scale = 2
```bash
CUDA_VISIBLE_DEVICES=0,1,4,5 python -u main.py --n_GPUs 4 --n_threads 8 \
--model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 --dilation 3 \
--save CFGN_CFGM_v2+ACT_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 128 --save_results --save_models --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0
```

Train CFGN with upsampling scale = 3
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py --n_GPUs 4 --n_threads 8 \
--model CFGN --scale 3 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 --dilation 3 --pre_train pre_train_model_path \
--save CFGN_CFGM_v2+ACT_BIX3_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 192 --save_results --save_models --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0
```

Train CFGN with upsampling scale = 4
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py --n_GPUs 4 --n_threads 8 \
--model CFGN --scale 4 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 --dilation 3 --pre_train pre_train_model_path \
--save CFGN_CFGM_v2+ACT_BIX4_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 256 --save_results --save_models --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 # --test_every 0
```

## Inference (Demo)
If you want to train CFGN, please ``cd */CFGN-PyTorch/src`` and run the following command in bash.

Test CFGN with upsampling scale = 2
```bash
CUDA_VISIBLE_DEVICES=2,3 python -u main.py --n_GPUs 2 --n_threads 4 \
--model CFGN --scale 2 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 --dilation 3 --pre_train pre_train_model_path \
--save CFGN_CFGM_v2+ACT_BIX2_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 128 --save_results --save_models --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 --test_only # --test_every 0
```

Test CFGN with upsampling scale = 3
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py --n_GPUs 4 --n_threads 8 \
--model CFGN --scale 3 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 --dilation 3 --pre_train pre_train_model_path \
--save CFGN_CFGM_v2+ACT_BIX3_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 192 --save_results --save_models --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 --test_only # --test_every 0
```

Test CFGN with upsampling scale = 4
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py --n_GPUs 4 --n_threads 8 \
--model CFGN --scale 4 --n_feats 64 --n_resgroups 9 --act lrelu --block_type CFGM_v2 --dilation 3 --pre_train pre_train_model_path \
--save CFGN_CFGM_v2+ACT_BIX4_F64R9 --data_test Set5+Set14+B100+Urban100+Manga109 --batch_size 64 --patch_size 256 --save_results --save_models --lr 0.0005 --decay 200-400-600-800-1000-1200 --epochs 0 --test_only # --test_every 0
```
