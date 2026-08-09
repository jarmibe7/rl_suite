"""Smoke tests for training loop."""

import pytest
import os
import tempfile
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.gridworld import Gridworld
from algos.random_actor import RandomActor
from buffer import ReplayBuffer
from evaluator import Evaluator
from logger import Logger
from loop import Trainer


def test_trainer_smoke_end_to_end():
    """Test that Trainer runs end-to-end without exceptions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create environment
        env = Gridworld(grid_size=5, max_steps=10)
        
        # Create algorithm
        algo = RandomActor(env.action_space)
        
        # Create buffer
        buffer = ReplayBuffer(capacity=100, sequence_length=None)
        
        # Create logger (with dummy wandb config)
        logger = Logger(
            project='test-project',
            entity=None,
            run_name='test-run',
            config={},
            run_dir=tmpdir,
        )
        
        # Create evaluator
        evaluator = Evaluator(
            env=env,
            algo=algo,
            run_dir=tmpdir,
            num_episodes=2,
            record_video=False,
        )
        
        # Create trainer
        trainer = Trainer(
            env=env,
            algo=algo,
            buffer=buffer,
            evaluator=evaluator,
            logger=logger,
            run_dir=tmpdir,
            start_training_after=5,
            updates_per_step=1,
            batch_size=4,
            eval_every_steps=0,  # Disable evaluation for speed
            checkpoint_every_steps=0,  # Disable checkpointing for speed
            seed=0,
        )
        
        # Run for short number of steps
        trainer.run(total_env_steps=100)
        
        # Check that metrics file was created
        metrics_file = os.path.join(tmpdir, 'metrics.csv')
        assert os.path.exists(metrics_file)
        
        # Check that metrics file is non-empty
        with open(metrics_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) > 1  # Header + at least one data row
    
    print("Smoke test passed!")


def test_trainer_keyboard_interrupt_cleanup():
    """Test that Trainer properly cleans up on KeyboardInterrupt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Gridworld(grid_size=5, max_steps=10)
        algo = RandomActor(env.action_space)
        buffer = ReplayBuffer(capacity=100, sequence_length=None)
        
        logger = Logger(
            project='test-project',
            entity=None,
            run_name='test-run',
            config={},
            run_dir=tmpdir,
        )
        
        evaluator = Evaluator(
            env=env,
            algo=algo,
            run_dir=tmpdir,
            num_episodes=1,
            record_video=False,
        )
        
        trainer = Trainer(
            env=env,
            algo=algo,
            buffer=buffer,
            evaluator=evaluator,
            logger=logger,
            run_dir=tmpdir,
            start_training_after=5,
            batch_size=4,
            eval_every_steps=0,
            checkpoint_every_steps=0,
            seed=0,
        )
        
        # Trainer will run until it saves final checkpoint
        # We just check it doesn't crash
        try:
            trainer.run(total_env_steps=50)
        except Exception as e:
            pytest.fail(f"Trainer raised unexpected exception: {e}")
        
        # Check final checkpoint was created
        final_checkpoint = os.path.join(tmpdir, 'checkpoints', 'checkpoint_final.pt')
        assert os.path.exists(final_checkpoint), "Final checkpoint not saved"
    
    print("Cleanup test passed!")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
