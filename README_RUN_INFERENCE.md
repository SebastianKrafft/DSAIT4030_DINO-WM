# Running DINO-WM Planning Inference

This guide covers:

- PointMaze inference with the official pretrained checkpoint.
- PointMaze inference with locally trained DINOv2 or DINOv3 checkpoints.
- Reproducing the paper's PointMaze planning parameters.
- Collecting success, distance, video, and runtime measurements.
- Adapting the same workflow to PushT.

All commands should be run from the repository root.

## 1. What Planning Inference Requires

Planning uses the following components:

1. A trained world-model predictor checkpoint.
2. The matching training `hydra.yaml`.
3. The corresponding frozen visual encoder, such as DINOv2 or DINOv3.
4. The environment dataset, which supplies validation trajectories,
   normalization statistics, and dataset goals when required.
5. The environment simulator.

Precomputed training embeddings are **not required** for online planning.
Planning renders observations and computes their visual features using the
encoder. If a checkpoint does not contain the frozen encoder weights,
`plan.py` instantiates the encoder described by its `hydra.yaml`.

## 2. Checkpoint Layout

`plan.py` expects this structure:

```text
<ckpt_base_path>/
└── outputs/
    └── <model_name>/
        ├── hydra.yaml
        └── checkpoints/
            └── model_latest.pth
```

`model_name` is the directory name inside `outputs/`, not the checkpoint
filename.

### Official checkpoints

The official DINO-WM checkpoints are available at:

https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28

For PointMaze:

```bash
mkdir -p pretrained_ckpts
unzip /path/to/outputs.zip "outputs/point_maze/*" -d pretrained_ckpts/
```

Expected files:

```text
pretrained_ckpts/outputs/point_maze/hydra.yaml
pretrained_ckpts/outputs/point_maze/checkpoints/model_latest.pth
```

For PushT, extract `outputs/pusht/*` instead.

### Locally trained checkpoints

A locally trained checkpoint must be placed or linked into the same layout.
Its `hydra.yaml` must describe the environment, dataset, visual encoder,
predictor architecture, frameskip, action/proprio encoders, and whether a
decoder exists.

Example:

```text
our_ckpts/
├── dinov2_model_best.pth
└── outputs/
    └── our_dinov2/
        ├── hydra.yaml
        └── checkpoints/
            └── model_latest.pth -> ../../../dinov2_model_best.pth
```

Do not use a PointMaze checkpoint for PushT. Action dimensions, proprioception,
dataset preprocessing, environment dynamics, and predictor training must match
the target environment.

## 3. Environment Setup

Activate the project environment:

```bash
conda activate dino_wm
```

### Dataset

Set `DATASET_DIR` to the parent directory containing the environment data:

```bash
export DATASET_DIR=/absolute/path/to/data
```

The checkpoint's `hydra.yaml` determines the exact subdirectory. In the
repository configs these are normally:

```text
$DATASET_DIR/point_maze
$DATASET_DIR/pusht_noise
```

The dataset is still required for `goal_source=random_state` because planning
loads normalization statistics and validation data.

### MuJoCo for PointMaze

PointMaze requires MuJoCo 2.1 and `mujoco_py`:

```bash
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
python -c "import mujoco_py; print('mujoco_py OK')"
```

On the 8 GB laptop GPU, force MuJoCo rendering onto the CPU. Otherwise, one
GPU renderer per evaluation environment can consume most GPU memory:

```bash
export MUJOCO_PY_FORCE_CPU=1
```

### Common runtime variables

```bash
export DATASET_DIR=/absolute/path/to/data
export WANDB_MODE=offline
export DISPLAY=:1
export MUJOCO_PY_FORCE_CPU=1
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export D4RL_SUPPRESS_IMPORT_ERROR=1
```

Use the active display returned by `echo $DISPLAY` if it is not `:1`.

## 4. PointMaze Paper Parameters

The PointMaze paper configuration is encoded in
`conf/plan_point_maze.yaml`:

| Parameter | Paper configuration |
|---|---:|
| Seed | 99 |
| Evaluation tasks | 50 |
| Goal source | `random_state` |
| Goal horizon | 5 |
| Planner | MPC with CEM |
| Planning horizon | 5 |
| Actions executed per MPC round | 5 |
| CEM samples per task and optimization step | 300 |
| CEM top-k | 30 |
| CEM optimization steps | 10 |
| Objective alpha | 0 |
| Objective mode | `last` |

The 50 evaluations are 50 distinct start/goal tasks. For every task and every
CEM optimization step, the planner evaluates 300 candidate action sequences.
One MPC round therefore evaluates:

```text
50 tasks x 300 candidates x 10 optimization steps
= 150,000 candidate action sequences
```

Each candidate has a horizon of five model actions.

The paper reports PointMaze success rates of:

| Planner | Paper success rate |
|---|---:|
| Open-loop CEM | 0.80 |
| MPC with CEM | 0.98 |

The paper does not report the repository's auxiliary state, visual, proprio,
or prediction-divergence metrics.

### Memory batching does not change CEM parameters

`sample_batch_size` only controls how many of the 300 candidates are processed
on the GPU simultaneously. All 300 candidates are still generated and one
global top-30 selection is performed.

On the RTX 5070 Laptop 8 GB, use:

```text
planner.sub_planner.sample_batch_size=10
```

This preserves the paper's 300 samples, top-30, and 10 optimization steps.
Larger GPUs may omit this override or use a larger batch.

## 5. Run PointMaze

The local commands below use `plan_local.yaml` to avoid the SLURM launcher in
`plan_point_maze.yaml`, then override it with the paper parameters.

### Official pretrained checkpoint

```bash
python plan.py --config-name plan_local.yaml \
  hydra.run.dir=plan_outputs/pretrained_model \
  ckpt_base_path=/absolute/path/to/DSAIT4030_DINO-WM/pretrained_ckpts \
  model_name=point_maze \
  seed=99 \
  n_evals=50 \
  goal_source=random_state \
  goal_H=5 \
  n_plot_samples=10 \
  objective.alpha=0 \
  objective.mode=last \
  planner.n_taken_actions=5 \
  planner.sub_planner.horizon=5 \
  planner.sub_planner.num_samples=300 \
  planner.sub_planner.topk=30 \
  planner.sub_planner.opt_steps=10 \
  planner.sub_planner.sample_batch_size=10
```

### Locally trained DINOv2 checkpoint

Only the checkpoint root and model name change:

```bash
python plan.py --config-name plan_local.yaml \
  hydra.run.dir=plan_outputs/our_dinov2 \
  ckpt_base_path=/absolute/path/to/DSAIT4030_DINO-WM/our_ckpts \
  model_name=our_dinov2 \
  seed=99 n_evals=50 goal_source=random_state goal_H=5 \
  n_plot_samples=10 objective.alpha=0 objective.mode=last \
  planner.n_taken_actions=5 \
  planner.sub_planner.horizon=5 \
  planner.sub_planner.num_samples=300 \
  planner.sub_planner.topk=30 \
  planner.sub_planner.opt_steps=10 \
  planner.sub_planner.sample_batch_size=10
```

For DINOv3, use a model directory whose `hydra.yaml` selects the matching
DINOv3 encoder:

```bash
hydra.run.dir=plan_outputs/our_dinov3
ckpt_base_path=/absolute/path/to/DSAIT4030_DINO-WM/our_ckpts
model_name=our_dinov3
```

The first DINOv3 run may need to download the encoder weights. Later runs use
the local model cache.

### Quick smoke test

Use a smoke test only to validate loading and execution:

```bash
python plan.py --config-name plan_local.yaml \
  hydra.run.dir=plan_outputs/smoke_test \
  seed=99 n_evals=1 n_plot_samples=1 \
  planner.max_iter=1 \
  planner.sub_planner.num_samples=10 \
  planner.sub_planner.topk=3 \
  planner.sub_planner.opt_steps=1 \
  planner.sub_planner.sample_batch_size=1
```

Smoke-test results must not be compared with the paper.

## 6. MPC Stopping Rule and Fair Comparisons

The public paper configs set:

```yaml
planner:
  max_iter: null
```

This means MPC keeps replanning until all tasks succeed. The paper reports
PointMaze MPC success of `0.98`, but it does not clearly document a finite MPC
round cap. Unlimited retries can make weak checkpoints run for a very long time
and can inflate eventual success.

For a controlled comparison between checkpoints, set the same finite cap for
every model, for example:

```bash
planner.max_iter=2
```

Report both the stopping rule and success after each MPC round. Do not compare
an unlimited-retry result for one checkpoint against a fixed-round result for
another.

### Reuse exactly the same evaluation targets

Every run saves `plan_targets.pkl`. Runs with the same seed and configuration
should generate the same targets. For the strongest paired comparison, reuse a
saved target file:

```bash
goal_source=file \
+goal_file_path=/absolute/path/to/reference_run/plan_targets.pkl
```

This removes any ambiguity about sampled start/goal pairs.

## 7. Output Files and Metrics

Each run directory contains:

```text
plan_targets.pkl
logs.json
runtime_metrics.json
runtime_metrics.csv
plan0.png, plan1.png, ...
plan0_*.mp4, plan1_*.mp4, ...
output_final.png
output_final_*.mp4
.hydra/config.yaml
.hydra/overrides.yaml
```

Important metrics in `logs.json`:

| Metric | Meaning |
|---|---|
| `success_rate` | Fraction of tasks satisfying the environment success rule |
| `mean_state_dist` | Mean final state distance to the goal |
| `mean_visual_dist` | Mean pixel-space distance to the goal observation |
| `mean_proprio_dist` | Mean proprioceptive distance to the goal |
| `mean_div_visual_emb` | Difference between predicted and realized visual embeddings |
| `mean_div_proprio_emb` | Difference between predicted and realized proprio embeddings |

For PointMaze, success means the agent is within 0.5 position units of the
goal. A success rate of `0.96` over 50 tasks means 48 succeeded and 2 failed.

Official checkpoints with a decoder produce imagined-versus-real
visualizations. Custom checkpoints without a decoder produce videos of the
real environment rollout beside the goal frame.

## 8. Runtime Measurements

Every new run automatically saves runtime measurements in:

```text
runtime_metrics.json
runtime_metrics.csv
```

The measurements are:

- **Inference:** one dynamics-predictor forward pass for one prediction step,
  batch size 32. Five warm-up runs and 20 measured runs are used.
- **Simulation rollout:** one simulator action step, batch size 1, excluding
  environment reset. One warm-up and three measured runs are used.
- **Planning:** CEM optimization time for each MPC iteration, excluding
  simulator evaluation, metric calculation, plotting, and video generation.
  Both total time and time per evaluation task are recorded.

CUDA is synchronized around GPU measurements.

### Paper Table 10 reference

| Metric | Paper time (s) |
|---|---:|
| Inference, batch 32 | 0.014 |
| Simulation rollout, batch 1 | 3.0 |
| Planning, CEM 100 samples x 10 steps | 53.0 |

For a planning-time workload matching Table 10, run a separate benchmark with:

```bash
n_evals=1 \
planner.max_iter=1 \
planner.sub_planner.num_samples=100 \
planner.sub_planner.topk=30 \
planner.sub_planner.opt_steps=10
```

The PointMaze result configuration uses `300 x 10`, so its planning time is not
directly comparable to the paper's `100 x 10` timing. Hardware, CUDA version,
encoder implementation, rendering backend, and CEM memory batch size must also
be reported with timing results.

## 9. PushT Changes

The same planning code supports PushT, but the checkpoint, dataset, goals,
objective, and CEM settings must be changed together.

PushT uses its own simulator rather than the PointMaze MuJoCo environment.
`MUJOCO_PY_FORCE_CPU` is therefore only relevant to PointMaze, while `DISPLAY`
may still be required for PushT rendering.

### PushT paper parameters

| Parameter | PushT configuration |
|---|---:|
| Seed | 99 |
| Evaluation tasks | 50 |
| Goal source | `dset` |
| Goal horizon | 5 |
| Planner | MPC with CEM |
| Planning horizon | 5 |
| Actions executed per MPC round | 5 |
| CEM samples | 300 |
| CEM top-k | 30 |
| CEM optimization steps | 30 |
| Objective alpha | 1 |
| Objective mode | `last` |

The important differences from PointMaze are:

- PushT uses goals sampled from dataset trajectories.
- PushT uses `objective.alpha=1`.
- PushT uses 30 CEM optimization steps instead of 10.
- A PushT-trained checkpoint and its matching `hydra.yaml` are mandatory.
- The PushT dataset must exist at the path referenced by that `hydra.yaml`,
  normally `$DATASET_DIR/pusht_noise`.

### PushT local command

Use `plan_local.yaml` with PushT overrides:

```bash
python plan.py --config-name plan_local.yaml \
  hydra.run.dir=plan_outputs/pusht_pretrained \
  ckpt_base_path=/absolute/path/to/checkpoint_root \
  model_name=pusht \
  seed=99 \
  n_evals=50 \
  goal_source=dset \
  goal_H=5 \
  n_plot_samples=10 \
  objective.alpha=1 \
  objective.mode=last \
  planner.n_taken_actions=5 \
  planner.sub_planner.horizon=5 \
  planner.sub_planner.num_samples=300 \
  planner.sub_planner.topk=30 \
  planner.sub_planner.opt_steps=30 \
  planner.sub_planner.sample_batch_size=10
```

For a custom PushT checkpoint, change `model_name` to its output-directory
name. Its training config must identify `env.name: pusht` and the corresponding
PushT dataset and encoder.

PushT's 30 optimization steps make it roughly three times heavier than the
PointMaze `300 x 10` CEM workload. Start with a one-task smoke test. If GPU
memory is insufficient, reduce only `sample_batch_size` first. Reducing
`num_samples` or `opt_steps` changes the scientific evaluation protocol.

## 10. What to Record in a Results Table

For every checkpoint, record:

- Environment and checkpoint name.
- Visual encoder version.
- Seed and target-file identity.
- Number of evaluation tasks.
- Goal source and goal horizon.
- MPC stopping rule or round count.
- CEM samples, top-k, optimization steps, and planning horizon.
- Success rate after each MPC round.
- Final distance metrics.
- Inference, simulation, and planning times.
- GPU model, memory, CUDA/PyTorch versions, and CEM memory batch size.
- Number and location of saved videos.

Only compare checkpoints when environment, target tasks, planner parameters,
and stopping rules are identical.
