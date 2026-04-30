# TODO: W&B Training Pipeline Setup

This file records the setup pipeline we followed for the shared stEVE training
tracking work.

## Done

- Created the Python environment plan:

```bash
conda create -n steve_training python=3.10
conda activate steve_training
```

- Created the shared W&B workspace/project location:

```text
https://wandb.ai/stEVE_training
```

- Added a W&B key field in `.env.example`.

The real API key should go in a local `.env` file, not in Git:

```bash
cp .env.example .env
```

- Added the W&B setup guide:

```text
stEVE_training/HOW_WANDB.md
```

- Added a requirements file for the training/W&B Python dependencies. The Torch
  entry is pinned to the CUDA 12.8 wheel so pip does not install a CUDA 13 build:

```bash
python -m pip install -r requirements.txt
```

Current requirements include:

- `gymnasium`
- `optuna`
- `torch`
- `numpy`
- `pillow`
- `pygame`
- `PyOpenGL`
- `pyyaml`
- `wandb`

## Next

- Install the requirements inside the `steve_training` conda environment.
- Install SOFA with `SofaPython3` and `BeamAdapter`.

SOFA is required before the training scripts can run the simulator. The error
`ModuleNotFoundError: No module named 'Sofa'` means this step is missing or the
SOFA environment variables are not set. See:

```text
stEVE_training/HOW_SOFA.md
```

- Install the local stEVE packages in that same environment:

```bash
python -m pip install -e ./eve
python -m pip install -e ./eve_bench
python -m pip install -e ./eve_rl
```

- Put the real W&B API key in `.env`.
- Run a short `--no_wandb` smoke test first, then run one online W&B test run:

Before training, run the environment sanity check and optional GIF capture:

```bash
python3 ./training_scripts/diagnostics/check_basicwirenav_env.py \
  --episodes 2 \
  --max_steps 100 \
  --policy naive_target \
  --gif diagnostics/basicwirenav_naive_target.gif
```

Then run one online W&B test run:

```bash
python3 ./training_scripts/BasicWireNav_train.py \
  -d cuda \
  -nw 2 \
  -n BasicWireNav_smoke \
  --heatup_steps 1000 \
  --training_steps 3000 \
  --eval_interval 1500 \
  --explore_episodes 1 \
  --eval_seed_count 2
```




# W&B Tracking

Training runs now log to the W&B workspace `stEVE_training` and project
`stEVE_trainin`.

## One-time setup

Install SOFA before running the training scripts. W&B can start without SOFA,
but the simulator will fail with `ModuleNotFoundError: No module named 'Sofa'`
until SOFA with `SofaPython3` and `BeamAdapter` is installed. See
`HOW_SOFA.md`.

Install W&B in the Python environment used for training:

```bash
python3 -m pip install -r requirements.txt
```

The requirements file pins PyTorch to the CUDA 12.8 wheel. If `nvidia-smi`
does not work on the machine, fix the NVIDIA driver first or run training with
`-d cpu`.

Create a local `.env` file in the repo root. This file is ignored by Git, so the
API key does not get committed:

```bash
cp .env.example .env
```

Edit `.env` and set:

```bash
WANDB_API_KEY=<your_wandb_api_key>
WANDB_ENTITY=stEVE_training
WANDB_PROJECT=stEVE_trainin
WANDB_GROUP=stEVE_training_pipeline
WANDB_MODE=online
```

I have created a custom shared team for collaborator, so you can use own W&B API key after accepting the invitation to
the `stEVE_training` workspace. Runs will appear under:

```text
https://wandb.ai/stEVE_training/stEVE_trainin
```

## Running training

Run the training scripts the same way as before:

```bash
python3 ./training_scripts/BasicWireNav_train.py -d cuda -nw 29 -n BasicWireNav
python3 ./training_scripts/ArchVariety_train.py -d cuda -nw 29 -n ArchVariety
python3 ./training_scripts/DualDeviceNav_train.py -d cuda -nw 29 -n DualDeviceNav
```

To disable W&B for a local test:

```bash
python3 ./training_scripts/BasicWireNav_train.py --no_wandb
```

For a short smoke test that should reach W&B metrics quickly, override the
default full-training schedule:

```bash
python3 ./training_scripts/BasicWireNav_train.py \
  -d cuda \
  -nw 2 \
  -n BasicWireNav_smoke \
  --heatup_steps 1000 \
  --training_steps 3000 \
  --eval_interval 1500 \
  --explore_episodes 1 \
  --eval_seed_count 2
```

Without those overrides, `BasicWireNav_smoke` is only the run name; the script
still uses the full default schedule.

The first explore/eval cycle can report `train/update_steps=0` because the
runner collects exploration before it has enough exploration steps to schedule
gradient updates. The smoke command above runs long enough to enter an update
cycle while keeping evaluation to two seeds.

To check the environment itself before training, run the diagnostic in
`HOW_ENV_CHECK.md`.

## What gets tracked

Each run logs the main RL configuration: device, worker count, learning rate,
hidden layers, embedder settings, batch size, replay buffer size, gamma, heatup
steps, training steps, evaluation interval, action repeat, and stochastic
evaluation mode.

During training, the W&B runner logs:

- `train/heatup_steps` and `train/heatup_episodes`
- `train/exploration_steps` and `train/exploration_episodes`
- `train/update_steps`
- `train/q1_loss`, `train/q2_loss`, and `train/policy_loss`
- `explore/episode_reward_mean`, `explore/episode_length_mean`,
  `explore/terminal_rate`, and `explore/truncation_rate`
- rollout environment averages under `explore/env/*`
- `eval/quality`, `eval/reward`, `eval/best_quality`
- `eval/steps` and `eval/episodes`
- evaluation environment metrics under `env/*` and `eval/env/*`

The most useful navigation metrics are:

- `target_distance_3d`: straight-line distance from the guidewire tip to target
- `target_distance_reduction` and `target_distance_reduction_ratio`: whether the
  tip is getting closer to the target
- `path_length_to_target`: remaining shortest centerline path from tip to target
- `path_length_reduction` and `path_length_reduction_ratio`: progress along the
  vessel path
- `insertion_length` and `insertion_fraction`: how far the instrument has been
  inserted
- `action_translation_abs_mean` and `action_rotation_abs_mean`: how aggressive
  the policy actions are
- `truncated_max_steps`, `truncated_vessel_end`, and `truncated_sim_error`: why
  episodes are ending

Local CSV results and checkpoints are still written to the existing `results/`
folders. W&B is an additional live tracking layer for comparing runs in the
shared workspace.






# Environment Sanity Check

Use this before long training runs to verify that `BasicWireNav` resets, accepts
actions, returns finite observations/rewards, and terminates or truncates within
the configured step limit.

## Run the Check

From the repo root:

```bash
conda activate steve_training
export SOFA_ROOT=$HOME/sofa_stEVE/install
export PYTHONPATH=$SOFA_ROOT/plugins/SofaPython3/lib/python3/site-packages:$PYTHONPATH

python3 ./training_scripts/diagnostics/check_basicwirenav_env.py \
  --episodes 2 \
  --max_steps 100 \
  --policy naive_target \
  --gif diagnostics/basicwirenav_naive_target.gif
```

The command prints:

- SOFA import status
- action space shape and action limits
- observation keys
- per-episode reward, step count, terminal/truncated status, success flag, path
  ratio, tip-to-target distance, path distance left, distance reduction, average
  translation speed, trajectory length, and action statistics

`CHECK PASSED` means the environment contract is healthy for a short rollout:
there were no `NaN` or infinite values, and every episode ended by either
`terminal=True` or `truncated=True` before the limit.

For a random/stupid policy, `truncated=True` with `success=False` is normal. It
means the episode timed out without reaching the target. `terminal=True` and
`success=True` means the target was reached.

## Policies

Available simple policies:

```text
zero          no movement
forward       forward insertion only
rotate        forward insertion plus constant rotation
wiggle        forward insertion plus oscillating rotation
random        random actions across the full action space
heatup        random actions using the same heatup range as training
naive_target  simple observation-based target steering
```

The `BasicWireNav` action shape is `(1, 2)`:

```text
[translation_mm_per_second, rotation_rad_per_second]
```

The default action limits are approximately:

```text
translation: -35 to 35 mm/s
rotation:    -3.14 to 3.14 rad/s
```

## Live Window

The GIF is usually more useful than the SOFA/Pygame window because it draws from
the actual normalized RL observations. If you still want a live SOFA window:

```bash
python3 ./training_scripts/diagnostics/check_basicwirenav_env.py \
  --episodes 1 \
  --max_steps 100 \
  --policy wiggle \
  --sofa_window
```

If the window is black but the diagnostic GIF and printed values update, the RL
environment can still be working.

## 3D Mesh GIF

To record the SOFA/OpenGL mesh renderer, use `--mesh_gif`. This opens the
SOFA/Pygame renderer and saves the returned frames:

```bash
python3 ./training_scripts/diagnostics/check_basicwirenav_env.py \
  --episodes 1 \
  --max_steps 80 \
  --policy wiggle \
  --mesh_gif diagnostics/basicwirenav_mesh_wiggle.gif \
  --mesh_gif_every 1
```

This is separate from `--gif`:

- `--gif` records the normalized RL observation: tracking points and target.
- `--mesh_gif` records the SOFA 3D mesh/OpenGL renderer.

## Training Smoke Test

For training, the first explore/eval cycle can show `update_steps=0` because the
runner collects exploration before it has accumulated enough steps for gradient
updates. To see updates sooner, run at least two exploration/evaluation cycles:

```bash
python3 ./training_scripts/BasicWireNav_train.py \
  -d cuda \
  -nw 2 \
  -n BasicWireNav_test \
  --heatup_steps 1000 \
  --training_steps 3000 \
  --eval_interval 1500 \
  --explore_episodes 1 \
  --eval_seed_count 2
```
