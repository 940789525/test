#!/bin/bash

# 测试ViT-B/32在MSVD数据集上的性能
job_name="xclip_msvd_vit32_test"
DATA_PATH="/home/wa24301158/dataset/MSVD"
MODEL_PATH="ckpts3/xclip_msvd_vit32_compress/pytorch_model.bin.0"

# 使用CUDA设备2,3进行评估
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    main_xclip.py --do_eval --num_thread_reader=4 \
    --batch_size=50 --n_display=25 \
    --data_path ${DATA_PATH} \
    --features_path ${DATA_PATH}/msvd_hevc \
    --output_dir ckpts3/${job_name} \
    --init_model ${MODEL_PATH} \
    --lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 96 \
    --datatype msvd \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0 --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 2>&1 | tee -a log/${job_name} 