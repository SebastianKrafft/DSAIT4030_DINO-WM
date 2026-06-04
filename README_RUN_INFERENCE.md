# Running DINO-WM Inference on PointMaze

This guide explains how to download the pretrained checkpoint, configure the planner, and run inference locally on a single GPU.

---

## 1. Download the Pretrained Checkpoint

The official DINO-WM checkpoints (PointMaze, PushT, Wall) are hosted on OSF:

**URL:** https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28

Under the `checkpoints/` folder, download **`outputs.zip`** (~908 MB).

Then extract only the PointMaze checkpoint into the repo (skip the other models and the `__MACOSX` junk):

```bash
cd /path/to/DSAIT4030_DINO-WM
mkdir -p pretrained_ckpts
unzip /path/to/outputs.zip "outputs/point_maze/*" -d pretrained_ckpts/
```

After extraction you should have:

```
pretrained_ckpts/
└── outputs/
    └── point_maze/
        ├── checkpoints/
        │   └── model_latest.pth    ← the model weights
        └── hydra.yaml              ← the training config (read by plan.py)
```

---

## 2. Prerequisites

### Mujoco
Mujoco 2.1 must be installed and on your `LD_LIBRARY_PATH`:

```bash
mkdir -p ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ~/.mujoco/
cd ~/.mujoco && tar -xzvf mujoco210-linux-x86_64.tar.gz
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
```

### mujoco_py system headers (Linux)
`mujoco_py` requires X11 and GL development headers to compile. Install them once:

```bash
sudo apt-get install -y libx11-dev libgl1-mesa-dev libglew-dev libglfw3-dev libosmesa6-dev patchelf
```

Then verify mujoco_py compiles:

```bash
conda activate dino_wm
python -c "import mujoco_py; print('OK', mujoco_py.__version__)"
```

### Dataset (for dataset-based goal sampling only)
If using `goal_source: dset` you also need the PointMaze dataset at `$DATASET_DIR/point_maze/`. For `goal_source: random_state` (the default below) the dataset is still loaded for normalisation stats — so it must be present either way.

```bash
export DATASET_DIR=/path/to/data   # must contain point_maze/
```

---

## 3. Configure `conf/plan_local.yaml`

Open `conf/plan_local.yaml` and set **`ckpt_base_path`** to the absolute path of the `pretrained_ckpts/` folder you created in step 1:

```yaml
ckpt_base_path: /absolute/path/to/DSAIT4030_DINO-WM/pretrained_ckpts
```

> **Important:** this must be an absolute path. Hydra changes the working directory at runtime, so relative paths will not resolve correctly.

The other values you may want to adjust:

| Key | Default | Meaning |
|-----|---------|---------|
| `model_name` | `point_maze` | Subfolder inside `ckpt_base_path/outputs/` |
| `model_epoch` | `latest` | `latest` loads `model_latest.pth`; set to an int for a specific epoch |
| `n_evals` | `3` | Number of episodes to evaluate |
| `goal_source` | `random_state` | `random_state` = random goals; `dset` = goals from dataset |
| `goal_H` | `5` | Steps ahead the goal is sampled from (for `dset` mode) |
| `n_plot_samples` | `3` | How many episodes to save as plots/videos |
| `planner.sub_planner.num_samples` | `100` | CEM candidate trajectories per step (reduce if OOM) |
| `planner.sub_planner.opt_steps` | `5` | CEM optimisation iterations per MPC step (reduce if OOM) |

### GPU memory note
The full paper config uses `num_samples: 300` and `opt_steps: 10`. On an 8 GB GPU use `num_samples: 100` and `opt_steps: 5` (already set as the default here) to avoid OOM.

---

## 4. Run Inference

```bash
cd /path/to/DSAIT4030_DINO-WM
conda activate dino_wm

export DATASET_DIR=/path/to/data
export WANDB_MODE=offline
export DISPLAY=:1    # set to your active X display (check with: echo $DISPLAY)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export D4RL_SUPPRESS_IMPORT_ERROR=1

python plan.py --config-name plan_local.yaml
```

---

## 5. Outputs

Results are saved to a timestamped folder under `plan_outputs/`:

```
plan_outputs/<timestamp>_point_maze_gH5/
├── output_final.png          ← side-by-side: real rollout vs world model prediction
├── output_final_0_success.mp4  ← video of episode 0 (maze navigation)
├── output_final_1_success.mp4
├── output_final_2_success.mp4
├── plan0.png / plan1.png     ← per-MPC-step trajectory comparisons
├── plan0_*.mp4               ← videos per MPC round
└── logs.json                 ← all metrics (success_rate, state_dist, visual_dist)
```

Open a video to see the robot navigating the maze:

```bash
xdg-open plan_outputs/<timestamp>_point_maze_gH5/output_final_0_success.mp4
```

Key metrics in `logs.json`:

| Metric | Meaning |
|--------|---------|
| `success_rate` | Fraction of episodes where robot reached within 0.5 units of goal |
| `mean_state_dist` | Average final distance to goal state |
| `mean_visual_dist` | Average visual embedding distance to goal observation |

---

## 6. Timing (on RTX 5070 Laptop 8 GB)

| Config | Episodes | Time |
|--------|----------|------|
| `num_samples=100`, `opt_steps=5` | 3 | ~2 min 11 s |
| `num_samples=300`, `opt_steps=10` (paper config) | 3 | ~6 min (estimated) |
| `num_samples=300`, `opt_steps=10` (paper config) | 50 | ~1–2 hours (estimated) |
