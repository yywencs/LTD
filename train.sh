#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py \
        --name=for_ltd_test \
        --wang2020_data_path="<data_path>" \
        --checkpoints_dir="<ckpt_dir>" \
        --data_mode="wang2020 or sd1_4"  \
        --arch="CLIP:ViT-L/14"  \
        --lr=0.00005 \
        --fix_backbone \
        --select_k=5 \
        --batch_size=256 \
        --niter=5
        
