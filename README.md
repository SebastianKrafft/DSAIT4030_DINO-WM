# DINO-WM: World Models on Pre-trained Visual Features (Reproduction & Extension)

This repository contains a streamlined and extended implementation of DINO-WM. Due to compute constraints, this project explicitly focuses on evaluating and comparing **DINOv2** and **DINOv3** visual encoders for zero-shot planning in the **PointMaze** and **Push-T** environments.

## 1. Repository Structure

This codebase has been deliberately pruned to focus only on the environments and architectures evaluated in our report. 

* `datasets/`: Data loading pipelines for PointMaze and Push-T.
* `env/`: Environment wrappers and simulator logic.
* `models/`: The core architectures (Frozen DINOv2/DINOv3 encoders, ViT transition model, and Transposed CNN decoder).
* `planning/`: MPC and Cross-Entropy Method (CEM) logic for zero-shot visual planning.
* `plan.py`: The primary entry point for running inference and generating evaluation videos.
* `conf/`: Hydra configuration files dictating architecture, dataset paths, and planning hyperparameters.

## 2. Installation

Set up the Conda environment:
```bash
conda env create -f environment.yaml
conda activate dino_wm
```

Install Mujoco (Required for PointMaze)
Create a .mujoco directory and download Mujoco210:
```bash
mkdir -p ~/.mujoco
wget [https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) -O mujoco.tar.gz
tar -xf mujoco.tar.gz -C ~/.mujoco
rm mujoco.tar.gz
```
Ensure you add the Mujoco path to your system variables (e.g., in ~/.bashrc):
```bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

## 3. Pre-Trained Checkpoints
Because training requires significant GPU compute hours, we have provided our final trained world model weights.

👉 Download Final Checkpoints (Google Drive): https://drive.google.com/drive/folders/1VaRqrkcHoplGAcE00edPJeMlyqpmbbLM?usp=drive_link

Download and extract this folder into the root directory of this repository. Do not alter the folder structure inside, as the Hydra configuration relies on this exact layout to align the architecture configs (hydra.yaml) with the weights (model_latest.pth).

Your project root should look like this:

```bash
DSAIT4030_DINO-WM/
├── final_checkpoints/
│   └── outputs/
│       ├── pointmaze_dinov2/
│       ├── pointmaze_dinov3/
│       ├── pointmaze_dinov3_repeat4/
│       └── pusht_dinov2/
└── plan.py
```

## 4. Running Inference (Planning)
Planning uses Model Predictive Control (MPC) with the Cross-Entropy Method (CEM) to optimize action sequences in the latent space. Precomputed embeddings are not required for online planning; the script will automatically instantiate the frozen visual encoder to process observations in real-time.

To launch planning, run the following commands from the repository root:

PointMaze Inference (Example: DINOv3)
This setup runs 50 evaluation tasks using random_state goals, with 300 CEM samples and 10 optimization steps per task.

```bash
python plan.py --config-name plan_local.yaml \
  ckpt_base_path=./final_checkpoints \
  env.dataset.data_path=./data/pointmaze \
  model_name=pointmaze_dinov3 \
  seed=99 n_evals=50 goal_source=random_state goal_H=5 \
  n_plot_samples=10 objective.alpha=0 objective.mode=last \
  planner.n_taken_actions=5 \
  planner.sub_planner.horizon=5 \
  planner.sub_planner.num_samples=300 \
  planner.sub_planner.topk=30 \
  planner.sub_planner.opt_steps=10 \
  planner.sub_planner.sample_batch_size=10
```
Push-T Inference (Example: DINOv2)
Push-T requires heavier optimization. This setup uses dataset-sampled goals (goal_source=dset), shifts the objective alpha to 1, and increases CEM optimization steps to 30.

Note: If GPU memory is insufficient, reduce planner.sub_planner.sample_batch_size before altering the core samples or opt_steps.
```bash
python plan.py --config-name plan_local.yaml \
  ckpt_base_path=./final_checkpoints \
  env.dataset.data_path=./data/pusht \
  model_name=pusht_dinov2 \
  seed=99 n_evals=50 goal_source=dset goal_H=5 \
  n_plot_samples=10 objective.alpha=1 objective.mode=last \
  planner.n_taken_actions=5 \
  planner.sub_planner.horizon=5 \
  planner.sub_planner.num_samples=300 \
  planner.sub_planner.topk=30 \
  planner.sub_planner.opt_steps=30 \
  planner.sub_planner.sample_batch_size=10
```

Evaluation Outputs
Planning logs, quantitative metrics (Success Rate, Final Distance), and visualization videos (.mp4 or .gif) are automatically saved to dynamically generated timestamped folders in the ./plan_outputs/ directory.