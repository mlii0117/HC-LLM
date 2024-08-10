#!/bin/bash

dataset="mimic_cxr"
annotation=/home/ltf/code/Longitudinal-Chest-X-Ray-main/
base_dir="/home/ltf/code/data/ALMM/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
delta_file="/home/ltf/code/R2GenGPT-HRRG6/save/mimic_cxr/v1_shallow/checkpoints/checkpoint_epoch0_step11546_bleu0.098568_cider0.265444.pth"

version="v1_shallow"
savepath="./save/$dataset/$version"

CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --test_batch_size 16 \
    --freeze_vm True \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 12 \
    --devices 1 \
    2>&1 |tee -a ${savepath}/log.txt