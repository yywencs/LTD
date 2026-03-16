#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python3 validate.py \
    --arch=CLIP:ViT-L/14  \
    --ckpt=<checkpoint_path> \
    --result_folder="results/" \
    --select_k=5 \
    --batch_size=256