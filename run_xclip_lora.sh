#!/bin/bash

# ViT-B/32
job_name="0607"
DATA_PATH="/root/autodl-tmp/MSVD/"
LOG_DIR="./log"
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/${job_name}.log"

# 确保 python -m torch.distributed.launch 后面的所有参数都属于 main_xclip.py
# 每个参数行尾的 \ 后面不能有任何空格

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    main_xclip.py \
    --do_train \
    --num_thread_reader=4 \
    --epochs=1 \
    --batch_size=60 \
    --n_display=5 \
    --data_path "${DATA_PATH}" \
    --features_path "${DATA_PATH}/msvd_hevc" \
    --output_dir "ckpts3/${job_name}" \
    --lr 1e-6 \
    --max_words 32 \
    --max_frames 12 \
    --batch_size_val 16 \
    --datatype msvd \
    --feature_framerate 1 \
    --coef_lr 1e-4 \
    --slice_framepos 2 \
    --loose_type \
    --linear_patch 2d \
    --sim_header meanP \
    --pretrained_clip_name "ViT-B/32" \
    --use_lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_target_modules 'clip.visual.transformer.resblocks' 'clip.transformer.resblocks' \
    --lora_bias 'none' \
    --fp16 > ${LOG_FILE} 2>&1