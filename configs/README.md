# Configuration Files

This directory contains YAML configuration files for training runs.

## File Format

Each config file specifies:
- **env**: Environment name and parameters
- **algo**: Algorithm name and parameters
- **training**: Training hyperparameters
- **evaluation**: Evaluation settings
- **checkpoint**: Checkpointing frequency
- **logging**: WandB logging configuration

## Available Environments

- `Gridworld`: Toy environment
- `PointGoal`: Continuous point-navigation environment
- `Pendulum` / `Pendulum-v1`: Classic control continuous swing-up
- `MountainCarContinuous` / `MountainCarContinuous-v0`: Classic control continuous hill climb

## Available Algorithms

- `random`: Random policy
- `SAC`: Soft Actor-Critic
- More algs coming soon !!! :3

## Example Configs

- `random_gridworld.yaml`: Random policy on Gridworld
- `sac_pendulum.yaml`: SAC on Pendulum
- ...

## Creating New Configs

Copy an existing config and modify parameters as needed. Remember to:
1. Check that algorithm `requires_sequences` matches buffer `sequence_length`
2. Ensure `batch_size` <= `buffer_capacity`
3. Set reasonable seeds for reproducibility
