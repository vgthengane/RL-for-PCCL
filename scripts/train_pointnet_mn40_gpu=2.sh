#!/bin/bash
#SBATCH --job-name=pointnet_mn40
#SBATCH --partition=3090_risk
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03-00:00:00
#SBATCH --output=/mnt/fast/nobackup/users/vt00262/projects/RL-for-PCCL/_experiments/slurm_logs/pointnet_mn40_%j.out
#SBATCH --error=/mnt/fast/nobackup/users/vt00262/projects/RL-for-PCCL/_experiments/slurm_logs/pointnet_mn40_%j.err

# srun --job-name=pointnet_mn40 --partition=3090_risk --gpus=2 --cpus-per-task=8 --mem=32G --time=04:00:00 --pty bash

WORK_DIR=/mnt/fast/nobackup/users/vt00262/projects/RL-for-PCCL/PointNet_v2
cd $WORK_DIR || exit 1

aprun exec torchrun \
    --nproc_per_node=2 \
    --master_port=29501 \
    PointNet_v2/train_classification.py \
        --model pointnet_cls \
        --num_category 40 \
        --epoch 200 \
        --learning_rate 0.001 \
        --num_point 1024 \
        --optimizer Adam \
        --log_dir pointnet_mn40_gpu=2 \
        --use_normals \
        --use_uniform_sample \
        --data_path /mnt/fast/nobackup/users/vt00262/projects/RL-for-PCCL/_datasets/modelnet40_normal_resampled