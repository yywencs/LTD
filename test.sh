#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 validate.py \
    --arch=CLIP:ViT-L/14  \
    --ckpt=<ckpt> \
    --result_folder="results/" \
    --select_k=5 \
    --batch_size=256