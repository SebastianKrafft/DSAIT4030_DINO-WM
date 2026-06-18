#!/bin/bash
#SBATCH --job-name=dino_v2_model_a100_8v_4g_64_fresh
#SBATCH --account=education-me-msc-ro
#SBATCH --partition=gpu-a100
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/home/%u/dino_wm_logs/train_v2_%j.out
#SBATCH --error=/home/%u/dino_wm_logs/train_v2_%j.err

mkdir -p ~/dino_wm_logs
mkdir -p ~/dino_wm_checkpoints/dino_v2_fresh

echo "=============================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "=============================="

module purge
module load 2024r1
module load miniforge3
module load cuda/12.1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate dino_wm

python -c "
import torch
props = torch.cuda.get_device_properties(0)
free, total = torch.cuda.mem_get_info()
print(f'GPU:        {props.name}')
print(f'VRAM total: {total/1024**3:.1f} GB')
print(f'VRAM free:  {free/1024**3:.1f} GB')
"

export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
export WANDB_MODE=online
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/dino_wm

python train.py \
	--config-name train.yaml \
	env=point_maze \
	env.dataset.data_path=/scratch/dtownsend/dino_wm_original/point_maze \
	env.dataset.embedding_dir=/scratch/dtownsend/dino_wm_data \
	env.dataset.n_rollout=2000 \
	env.num_workers=8 \
	frameskip=5 \
	num_hist=3 \
	training.batch_size=64 \
	has_decoder=False \
	training.epochs=100 \
	training.save_every_x_epoch=50 \
	predictor.use_flash_attention=true \
	ckpt_base_path=/home/$USER/dino_wm_checkpoints/dino_v2_fresh \
	hydra.run.dir=/home/$USER/dino_wm_checkpoints/dino_v2_fresh

echo "=============================="
echo "Job finished: $(date)"
echo "=============================="


