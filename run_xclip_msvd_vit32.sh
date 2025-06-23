# ViT-B/32
job_name="xclip_msvd_vit32"
DATA_PATH="/home/wa24301158/dataset/MSVD"
# python -m torch.distributed.launch --nproc_per_node=4 \
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    main_xclip.py --do_train --num_thread_reader=1 \
    --epochs=5 --batch_size=50 --n_display=10 \
    --data_path ${DATA_PATH}/MSVD \
    --features_path ${DATA_PATH}/MSVD_Videos \
    --output_dir ckpts3/${job_name} \
    --lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 96 \
    --datatype msvd \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0 --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 2>&1 | tee -a log/${job_name}
