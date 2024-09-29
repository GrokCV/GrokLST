cd /data1/dym/GrokLST-Dev/
CUDA_VISIBLE_DEVICES=1,2,3,4
/data1/dym/anaconda3/envs/groklst/bin/python ./tools/dist_train.sh ./configs/mocolsk/mocolsk_x2_4xb1-10k_lst.py 4
