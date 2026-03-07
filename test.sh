CUDA_VISIBLE_DEVICES=3 python3 validate.py \
    --arch=CLIP:ViT-L/14  \
    --ckpt="/mnt/c1a908c8-4782-4898-8be7-c6be59a325d6/yyw/detection_task/origin_delta/dual_branch/Ojha/k_5_11_20_5e-5/model_epoch_5.pth" \
    --result_folder="results/UFD/" \
    --select_k=5 \
    --batch_size=32