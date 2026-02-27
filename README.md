# Coursework 2: On-Policy vs Off-Policy Reinforcement Learning for Continuous Control

This assignment asks you to compare **PPO** (Proximal Policy Optimization, on-policy) and **SAC** (Soft Actor-Critic, off-policy) on a MuJoCo continuous-control task. You will implement the core update steps in each algorithm, tune hyperparameters via a grid search, run multi-seed experiments, and write a report analysing the results.

---

## Repository Structure

```
agents/
  ppo.py                # PPO skeleton — implement 4 stub functions
  sac.py                # SAC skeleton — implement 4 stub functions
notebooks/
  plot_results.ipynb    # plot learning curves from output/
  render_policy.ipynb   # load a saved model.pt and render the policy
output/                 # training artefacts written here automatically
```

---

## Prerequisites & Installation

**Python 3.10 is required** (`>=3.10,<3.11`).

**We recommend you use UV for environment mangagement:** [See UV installatin guide](https://docs.astral.sh/uv/getting-started/installation/)

```bash
git clone git@github.com:LeoHink/rrl_mujoco_cw2.git
```

```bash
cd rrl_mujoco_cw2
uv sync 
```
or (either should work)

```bash
uv pip install .
uv pip install ".[mujoco]"   # MuJoCo environments
```

---

## What Is Provided vs What You Must Implement

### Provided (skeleton)

- Environment wrappers, training loops, logging, and model saving
- Actor-critic networks (PPO) and Actor + twin Q-networks (SAC)
- Replay buffer and observation normalisation
- Plotting and rendering notebooks

### You Must Implement

The functions below raise `NotImplementedError`. Fill them in.

**`agents/ppo.py`**

| Function | What it computes |
|---|---|
| `compute_gae` | GAE advantages and return targets (backward TD recursion) |
| `compute_ratio` | log-ratio and ratio π_θ / π_old |
| `compute_policy_loss` | PPO clipped surrogate objective |
| `compute_value_loss` | MSE value-function loss |

**`agents/sac.py`**

| Function | What it computes |
|---|---|
| `compute_q_target` | Bellman TD target: r + γ(1−d) min Q_target |
| `compute_actor_loss` | Policy loss: E[α log π − min Q] |
| `compute_alpha_loss` | Entropy coefficient tuning loss |
| `soft_update` | Polyak average θ_target ← τθ + (1−τ)θ_target |

---

## Running the Agents

```bash
# PPO
python agents/ppo.py --env-id HalfCheetah-v4 --seed 1 --total-timesteps 1000000

# SAC
python agents/sac.py --env-id Hopper-v4 --seed 1 --total-timesteps 1000000
```

All hyperparameters are CLI flags (via `tyro`). Use `--help` to see all options.

Outputs are saved automatically to `output/{env_id}/{algorithm}/{timestamp}_seed{seed}/`:

```
output/{env_id}/{algorithm}/{timestamp}_seed{seed}/
├── config.yaml     # all hyperparameters used
├── returns.npy     # episode returns array
└── model.pt        # network weights (+ obs_rms for PPO)
```

### Final run

Once you have chosen your best hyperparameters, record the final multi-seed results in a
stable, predictable directory by passing `--final-run`:

```bash
# PPO — repeat for each seed
python agents/ppo.py --env-id HalfCheetah-v4 --seed 1 --final-run --clip-coef 0.2 --learning-rate 3e-4
python agents/ppo.py --env-id HalfCheetah-v4 --seed 2 --final-run ...

# SAC — repeat for each seed
python agents/sac.py --env-id Hopper-v4 --seed 1 --final-run --tau 0.005 --q-lr 1e-3
python agents/sac.py --env-id Hopper-v4 --seed 2 --final-run ...
```

Outputs land in `output/{env_id}/{algorithm}/final_run/seed{seed}/` — no timestamp, easy
to find and load in `notebooks/plot_results.ipynb`.

### Available MuJoCo Environments

Pick one environment to focus on for your experiments.

| `--env-id` | Description |
|---|---|
| `Ant-v4` | 8-DoF ant locomotion |
| `HalfCheetah-v4` | Planar cheetah running |
| `Hopper-v4` | One-legged hopping |
| `Humanoid-v4` | 17-DoF humanoid locomotion |
| `HumanoidStandup-v4` | Humanoid standing up |
| `InvertedDoublePendulum-v4` | Balance double pendulum |
| `Pusher-v4` | Arm pushing object to goal |
| `Reacher-v4` | 2-DoF arm reaching |
| `Swimmer-v4` | 3-link swimmer |
| `Walker2d-v4` | Bipedal walker |

---

## Notebooks

You can run these directly through the standard Jupyter Notebook interface or you can run these in VS Code with extension. 

### Running Notebooks in VS Code

Open `notebooks/plot_results.ipynb` or `notebooks/render_policy.ipynb` directly in VS Code and select the `.venv` kernel from the kernel picker.

#### Kernel not showing up in VS Code?

If the `.venv` kernel does not appear in the kernel picker, start a Jupyter server manually:

```bash
uv run jupyter notebook --no-browser
```

The terminal will print a URL like:

```
http://localhost:8888/?token=<your-token>
```

In VS Code, you can connect your running jupyter server by selecting a Kernel and opting to **"Existing Jupyter server ..."**, and paste the URL. VS Code will connect to that server and the correct kernel will be available.:w

---

## Hyperparameter Search & Multi-Seed Runs

Run a **3×3 grid search** — 3 values for each of 2 hyperparameters of your choosing (e.g. learning rate and clipping coefficient for PPO; learning rate and `tau` for SAC). Note in the interest of time it is completely fine to run just a single seed for the sweeps. Results are saved automatically; use `notebooks/plot_results.ipynb` to generate learning-curve figures. Evaluate and find the best hyperparameter configuration.

---



