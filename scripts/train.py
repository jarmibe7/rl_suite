"""Training entrypoint script."""

import argparse
import yaml
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import torch
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.wrappers import make_env
from algos.random_actor import RandomActor
from buffer import ReplayBuffer
from evaluator import Evaluator
from logger import Logger
from loop import Trainer


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Environment
    env_name: str
    env_params: Dict[str, Any]
    
    # Algorithm
    algo_name: str
    algo_params: Dict[str, Any]
    
    # Training
    total_steps: int
    batch_size: int
    buffer_capacity: int
    buffer_sequence_length: Optional[int] = None
    start_training_after: int = 1000
    updates_per_step: int = 1
    seed: int = 0
    
    # Evaluation
    eval_every_steps: int = 1000
    num_eval_episodes: int = 5
    record_video: bool = True
    video_every_calls: int = 1
    
    # Checkpointing
    checkpoint_every_steps: int = 10000
    
    # Logging
    wandb_project: str = 'rl-suite'
    wandb_entity: Optional[str] = None


# Algorithm registry
ALGO_REGISTRY = {
    'random': RandomActor,
}


def load_config(config_path: str) -> TrainingConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        TrainingConfig instance
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract sections
    env_config = config_dict.get('env', {})
    algo_config = config_dict.get('algo', {})
    training_config = config_dict.get('training', {})
    eval_config = config_dict.get('evaluation', {})
    checkpoint_config = config_dict.get('checkpoint', {})
    logging_config = config_dict.get('logging', {})
    
    # Build TrainingConfig
    config = TrainingConfig(
        # Environment
        env_name=env_config.get('name', 'Gridworld'),
        env_params=env_config.get('params', {}),
        
        # Algorithm
        algo_name=algo_config.get('name', 'random'),
        algo_params=algo_config.get('params', {}),
        
        # Training
        total_steps=training_config.get('total_steps', 10000),
        batch_size=training_config.get('batch_size', 32),
        buffer_capacity=training_config.get('buffer_capacity', 50000),
        buffer_sequence_length=training_config.get('buffer_sequence_length', None),
        start_training_after=training_config.get('start_training_after', 1000),
        updates_per_step=training_config.get('updates_per_step', 1),
        seed=training_config.get('seed', 0),
        
        # Evaluation
        eval_every_steps=eval_config.get('eval_every_steps', 1000),
        num_eval_episodes=eval_config.get('num_episodes', 5),
        record_video=eval_config.get('record_video', True),
        video_every_calls=eval_config.get('video_every_calls', 1),
        
        # Checkpointing
        checkpoint_every_steps=checkpoint_config.get('checkpoint_every_steps', 10000),
        
        # Logging
        wandb_project=logging_config.get('wandb_project', 'rl-suite'),
        wandb_entity=logging_config.get('wandb_entity', None),
    )
    
    return config


def validate_config(config: TrainingConfig) -> None:
    """Validate configuration for consistency.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check algrorithm exists
    if config.algo_name not in ALGO_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{config.algo_name}'. "
            f"Available: {list(ALGO_REGISTRY.keys())}"
        )
    
    # Check algorithm and buffer sequence mode compatibility
    algo_class = ALGO_REGISTRY[config.algo_name]
    algo_requires_sequences = getattr(algo_class, 'requires_sequences', False)
    buffer_has_sequences = config.buffer_sequence_length is not None
    
    if algo_requires_sequences and not buffer_has_sequences:
        raise ValueError(
            f"Algorithm '{config.algo_name}' requires sequences "
            f"(requires_sequences=True), but buffer.sequence_length is None"
        )
    
    if not algo_requires_sequences and buffer_has_sequences:
        print(
            f"Warning: Algorithm '{config.algo_name}' does not require sequences, "
            f"but buffer.sequence_length={config.buffer_sequence_length}. "
            f"This is okay but inefficient."
        )
    
    # Check training parameters
    if config.batch_size > config.buffer_capacity:
        raise ValueError(
            f"batch_size ({config.batch_size}) cannot exceed "
            f"buffer_capacity ({config.buffer_capacity})"
        )
    
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.batch_size}")
    
    if config.total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {config.total_steps}")


def generate_run_dir(algo_name: str, base_dir: str = 'runs') -> str:
    """Generate a timestamped run directory.
    
    Args:
        algo_name: Algorithm name
        base_dir: Base directory for runs
        
    Returns:
        Path to run directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_dir, algo_name, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def build_and_train(config: TrainingConfig, run_dir: str) -> None:
    """Build components and run training.
    
    Args:
        config: Training configuration
        run_dir: Directory to save run artifacts
    """
    print(f"\n{'='*60}")
    print(f"Starting training run")
    print(f"{'='*60}")
    print(f"Run directory: {run_dir}")
    print(f"Algorithm: {config.algo_name}")
    print(f"Environment: {config.env_name}")
    print(f"Total steps: {config.total_steps}")
    print(f"{'='*60}\n")
    
    # Save resolved config
    config_save_path = os.path.join(run_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump({
            'env': {
                'name': config.env_name,
                'params': config.env_params,
            },
            'algo': {
                'name': config.algo_name,
                'params': config.algo_params,
            },
            'training': {
                'total_steps': config.total_steps,
                'batch_size': config.batch_size,
                'buffer_capacity': config.buffer_capacity,
                'buffer_sequence_length': config.buffer_sequence_length,
                'start_training_after': config.start_training_after,
                'updates_per_step': config.updates_per_step,
                'seed': config.seed,
            },
            'evaluation': {
                'eval_every_steps': config.eval_every_steps,
                'num_episodes': config.num_eval_episodes,
                'record_video': config.record_video,
                'video_every_calls': config.video_every_calls,
            },
            'checkpoint': {
                'checkpoint_every_steps': config.checkpoint_every_steps,
            },
        }, f)
    print(f"Config saved to {config_save_path}")
    
    # Build environment
    print("\nBuilding environment...")
    env = make_env(config.env_name, seed=config.seed, **config.env_params)
    print(f"Environment observation shape: {env.observation_space.shape}")
    print(f"Environment action shape: {env.action_space.shape}")
    
    # Build algorithm
    print("\nBuilding algorithm...")
    algo_class = ALGO_REGISTRY[config.algo_name]
    algo = algo_class(env.action_space, **config.algo_params)
    print(f"Algorithm created: {config.algo_name}")
    
    # Build buffer
    print("\nBuilding replay buffer...")
    buffer = ReplayBuffer(
        capacity=config.buffer_capacity,
        sequence_length=config.buffer_sequence_length,
    )
    print(
        f"Buffer: capacity={config.buffer_capacity}, "
        f"mode={'sequence' if buffer.sequence_mode else 'transition'}"
    )
    
    # Build logger
    print("\nBuilding logger...")
    logger = Logger(
        project=config.wandb_project,
        entity=config.wandb_entity,
        run_name=os.path.basename(run_dir),
        config={k: v for k, v in vars(config).items()},
        run_dir=run_dir,
    )
    print(f"Logger initialized (project: {config.wandb_project})")
    
    # Build evaluator
    print("\nBuilding evaluator...")
    evaluator = Evaluator(
        env=env,
        algo=algo,
        run_dir=run_dir,
        num_episodes=config.num_eval_episodes,
        record_video=config.record_video,
        video_every_calls=config.video_every_calls,
    )
    print(f"Evaluator initialized (num_episodes={config.num_eval_episodes})")
    
    # Build trainer
    print("\nBuilding trainer...")
    trainer = Trainer(
        env=env,
        algo=algo,
        buffer=buffer,
        evaluator=evaluator,
        logger=logger,
        run_dir=run_dir,
        start_training_after=config.start_training_after,
        updates_per_step=config.updates_per_step,
        batch_size=config.batch_size,
        eval_every_steps=config.eval_every_steps,
        checkpoint_every_steps=config.checkpoint_every_steps,
        seed=config.seed,
    )
    print("Trainer initialized")
    
    # Run training
    print("\nStarting training loop...\n")
    trainer.run(config.total_steps)
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Results saved to: {run_dir}")
    print(f"{'='*60}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train an RL algorithm')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file',
    )
    parser.add_argument(
        '--run-dir',
        type=str,
        default=None,
        help='Override run directory (default: auto-generated timestamp)',
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    # Validate config
    print("Validating config...")
    validate_config(config)
    
    # Generate or use provided run directory
    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = generate_run_dir(config.algo_name)
    
    # Build and train
    build_and_train(config, run_dir)


if __name__ == '__main__':
    main()
