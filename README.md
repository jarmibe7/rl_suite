# RL Suite

A modular, algorithm-agnostic reinforcement learning training framework.

## Overview

RL Suite is designed to:
- **Separate concerns**: The training loop, buffer, evaluation, and logging are algorithm-agnostic
- **Enable experimentation**: Easy to swap algorithms and environments without touching core infrastructure
- **Minimize bloat**: Focused design with only what's needed for the use case
- **Use Wandb for logging**: Integrated Wandb support for live plotting and logging

## Architecture

```
rl_suite/
├── algos/                 # Algorithm implementations
├── buffer.py              # Replay buffer
├── loop.py                # Training loop
├── evaluator.py           # Evaluation utils
├── logger.py              # WandB logging wrapper
├── envs/
│   ├── gridworld.py       # Gridworld and point-goal environments
│   └── wrappers.py        # Environment standardization
├── configs/               # YAML configurations
├── scripts/
│   └── train.py           # CLI entrypoint
└── tests/                 # Test suite
```

## Quick Start

### Installation
First make a virtual environment. Then:

```bash
pip install -r requirements.txt
```

### Training

```bash
python scripts/train.py --config configs/<alg_name>/<config_name>.yaml
```

Results are saved to `runs/<algo>/<env>/<timestamp>/`.

## Components

### Algorithm (ABC)

All algorithms inherit from `Algorithm` and implement:
- `act(obs, deterministic=False)`: Select action
- `update(batch)`: Update parameters, return metrics dict
- `save/load(path)`: Checkpoint management
- `reset()`: Reset internal state (RNNs, etc.)
- `requires_sequences`: Boolean flag for sequence vs. transition buffer mode

### ReplayBuffer

Supports two modes:
- **Transition mode** (`sequence_length=None`): Individual (s, a, r, s', done) tuples
- **Sequence mode** (`sequence_length=L`): Full episodes stored in (T, B, ...) chunks

### Trainer

Main loop that:
1. Steps environment with `algo.act(obs)`
2. Stores transitions/episodes in buffer
3. Logs episode returns and lengths
4. Samples from buffer and updates algorithm
5. Evaluates periodically
6. Checkpoints periodically
7. Handles cleanup on exit or exception

### Evaluator

Runs deterministic rollouts to evaluate current policy without updating weights.
Supports video recording (with opencv).

### Logger

Wrapper around Wandb for live experiment tracking.
All metrics logged with `train/` and `eval/` prefixes.

## Configuration

Configs are YAML files with sections:
- `env`: Environment name and parameters
- `algo`: Algorithm name and parameters
- `training`: Batch size, buffer capacity, total steps, etc.
- `evaluation`: Evaluation frequency and settings
- `checkpoint`: Save frequency
- `logging`: WandB project and entity

See `configs/sac/sac_pointgoal.yaml` for an example.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run training without logging
WANDB_MODE=offline python scripts/train.py --config configs/<alg_name>/<config_name>.yaml
```

## Extending

### Adding a New Algorithm

1. Create `algos/my_algo.py`:
   ```python
   from algos.base import Algorithm
   
   class MyAlgo(Algorithm):
       requires_sequences = False  # or True
       
       def act(self, obs, deterministic=False):
           ...
       
       def update(self, batch):
           # Return dict of metrics
           return {'loss': 0.5, 'entropy': 0.3, ...}
       
       def save(self, path):
           ...
       
       def load(self, path):
           ...
   ```

2. Register in `scripts/train.py`:
   ```python
   ALGO_REGISTRY = {
       'my_algo': MyAlgo,
       ...
   }
   ```

3. Create config in `configs/my_algo/my_algo_*.yaml`

4. Run: `python scripts/train.py --config configs/my_algo/my_algo_*.yaml`

### Adding a New Environment

1. Implement Gymnasium-compatible environment or wrap existing
2. Use `envs/wrappers.py` to standardize interface
3. Register in `make_env()` in `wrappers.py` or add a Gymnasium alias there
4. Create config in `configs/*.yaml`

### Built-in Environment Names

- `Gridworld`
- `PointGoal`
- `Pendulum` or `Pendulum-v1`
- `MountainCarContinuous` or `MountainCarContinuous-v0`

## Notes

Github Copilot was used to assist in the production of this code.
