#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python train.py \
        --name=rz_sd14_genImage \
        --wang2020_data_path="data_path" \
        --checkpoints_dir="<checkpoint_path>" \
        --data_mode="wang2020 or sd1_4"  \
        --arch="CLIP:ViT-L/14"  \
        --lr=0.00001 \
        --fix_backbone \
        --select_k=5 \
        --batch_size=256 \
        --niter=5 \
        --save_epoch_freq=1
        
