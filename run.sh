####################  GM_MODILE gconv.py topK ####################

export CUDA_VISIBLE_DEVICES=0,1,2
nohup python3 -u train.py --train --bsize 256 --data_type VQA --data_dir ../data/VQA --save_dir ./trained_model/VQA_qg11_BiGRU2048_vqqemb_GE2_norm_GA2_norm_GMA2_max_b256_BEC2_ep35_25_lr_dacay05_mask >./log/log_VQA_qg11_BiGRU2048_vqqemb_GE2_norm_GA2_norm_GMA2_max_b256_BEC2_ep35_25_lr_decay05_mask.txt 2>&1 &

#export CUDA_VISIBLE_DEVICES=2,3
#nohup python3 -u train.py --trainval --bsize 256 --data_type VQA --data_dir ../data/VQA --save_dir ./trained_model/VQA_qg11_trainval_augvg_trainval_BiGRU2048_allp_tv3129a_vqqemb_GE2_norm_GA2_norm_GMA2_max_BEC2_b256_ep35_25_lr_dacay05 >./log/log_VQA_qg11_trainval_augvg_trainval_BiGRU2048_allp_tv3129a_vqqemb_GE2_norm_GA2_norm_GMA2_max_BEC2_b256_ep35_25_lr_decay05.txt 2>&1 &


# allp: train + val + VG

#export CUDA_VISIBLE_DEVICES=5
#nohup python3 -u train.py --trainval --bsize 256 --data_type VQA --data_dir ../data/VQA --save_dir ./trained_model/VQA_trainval_qg9_allp_tv3129a_augdata_100_BiGRU2048_vqqemb_S1_soft2_max_lf3_b256_ep30_lrdecay05 >./log/log_VQA_trainval_qg9_allp_tv3129a_augdata_100_BiGRU2048_vqqemb_S1_soft2_max_lf3_b256_ep30_lrdecay05.txt 2>&1 &


## eval

#export CUDA_VISIBLE_DEVICES=3
#python3 -u train.py --eval --model_path ./trained_model/GQA_vqu_vqqemb_S1_relu_soft2_max_drop_vq025_w03_f05_grad_relu1_lf3_Exlr09/model_23.pth.tar --data_type GQA --data_dir ../data/GQA --bsize 200 --neighbourhood_size 4
#
#export CUDA_VISIBLE_DEVICES=4,5
#python3 -u train.py --test --model_path ./trained_model/VQA_qg11_trainval_augvg_trainval_BiGRU2048_allp_tv3129a_vqqemb_GE2_norm_GA2_norm_GMA2_max_BEC2_b256_ep35_25_lr_dacay05/model_35.pth.tar --data_type VQA --data_dir ../data/VQA --bsize 256









