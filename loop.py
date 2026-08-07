"""Main training loop."""

import os
import csv
import torch
import numpy as np
from typing import Optional
from tqdm import tqdm
from datetime import datetime


class Trainer:
    """Main training loop. The only class that calls env.step().
    
    Coordinates environment interaction, buffer management, algorithm updates,
    evaluation, and logging.
    """
    
    def __init__(
        self,
        env,
        algo,
        buffer,
        evaluator,
        logger,
        run_dir: str,
        start_training_after: int = 1000,
        updates_per_step: int = 1,
        batch_size: int = 32,
        eval_every_steps: int = 1000,
        checkpoint_every_steps: int = 10000,
        seed: int = 0,
    ):
        """Initialize trainer.
        
        Args:
            env: Gymnasium environment
            algo: Algorithm instance (inherits from Algorithm)
            buffer: ReplayBuffer instance
            evaluator: Evaluator instance
            logger: Logger instance
            run_dir: Directory to save checkpoints and logs
            start_training_after: Number of steps before starting training
            updates_per_step: Number of weight update steps per environment step
            batch_size: Batch size for training
            eval_every_steps: Evaluate every N steps
            checkpoint_every_steps: Save checkpoint every N steps
            seed: Random seed
        """
        self.env = env
        self.algo = algo
        self.buffer = buffer
        self.evaluator = evaluator
        self.logger = logger
        self.run_dir = run_dir
        
        self.start_training_after = start_training_after
        self.updates_per_step = updates_per_step
        self.batch_size = batch_size
        self.eval_every_steps = eval_every_steps
        self.checkpoint_every_steps = checkpoint_every_steps
        
        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Create checkpoints directory
        self.checkpoint_dir = os.path.join(run_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Metrics tracking
        self.metrics_csv_path = os.path.join(run_dir, 'metrics.csv')
        self.metrics_file = None
        self.metrics_writer = None
        self._init_metrics_csv()
        
        self.global_step = 0
        self.episode_count = 0
    
    def run(self, total_env_steps: int) -> None:
        """Run training loop.
        
        Args:
            total_env_steps: Total number of environment steps to run
        """
        try:
            with tqdm(total=total_env_steps, desc='Training') as pbar:
                obs, _ = self.env.reset()
                self.algo.reset()
                
                episode_return = 0.0
                episode_length = 0
                
                while self.global_step < total_env_steps:
                    # Step environment
                    action = self.algo.act(obs, deterministic=False)
                    next_obs, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    
                    episode_return += float(reward)
                    episode_length += 1
                    
                    # Store in buffer
                    if self.buffer.sequence_mode:
                        # Accumulate step for add_episode at episode end
                        if not hasattr(self, '_episode_data'):
                            self._episode_data = {
                                'obs': [],
                                'action': [],
                                'reward': [],
                                'done': [],
                            }
                        
                        self._episode_data['obs'].append(obs)
                        self._episode_data['action'].append(action)
                        self._episode_data['reward'].append(reward)
                        self._episode_data['done'].append(done)
                    else:
                        # Transition mode: add immediately
                        self.buffer.add_step(obs, action, reward, next_obs, done)
                    
                    # Episode management
                    if done:
                        # Log episode metrics
                        self.logger.log({
                            'train/episode_return': episode_return,
                            'train/episode_length': episode_length,
                        }, step=self.global_step)
                        
                        # Write to CSV
                        self._write_metrics({
                            'step': self.global_step,
                            'episode': self.episode_count,
                            'episode_return': episode_return,
                            'episode_length': episode_length,
                        })
                        
                        # Add episode to buffer if sequence mode
                        if self.buffer.sequence_mode:
                            episode_dict = {
                                'obs': np.array(self._episode_data['obs']),
                                'action': np.array(self._episode_data['action']),
                                'reward': np.array(self._episode_data['reward']),
                                'done': np.array(self._episode_data['done']),
                            }
                            self.buffer.add_episode(episode_dict)
                            self._episode_data = None
                        
                        self.episode_count += 1
                        
                        # Reset for next episode
                        obs, _ = self.env.reset()
                        self.algo.reset()
                        episode_return = 0.0
                        episode_length = 0
                    else:
                        obs = next_obs
                    
                    self.global_step += 1
                    pbar.update(1)
                    
                    # Training updates
                    if (self.global_step >= self.start_training_after and
                        self.buffer.ready(self.batch_size)):
                        for _ in range(self.updates_per_step):
                            batch = self.buffer.sample(self.batch_size)
                            metrics = self.algo.update(batch)
                            
                            # Log training metrics
                            if metrics:
                                prefixed_metrics = {
                                    f'train/{k}': v for k, v in metrics.items()
                                }
                                self.logger.log(prefixed_metrics, step=self.global_step)
                    
                    # Evaluation
                    if (self.eval_every_steps > 0 and
                        self.global_step % self.eval_every_steps == 0):
                        eval_metrics = self.evaluator.run(self.global_step)
                        prefixed_eval = {
                            f'eval/{k}': v for k, v in eval_metrics.items()
                        }
                        self.logger.log(prefixed_eval, step=self.global_step)
                    
                    # Checkpointing
                    if (self.checkpoint_every_steps > 0 and
                        self.global_step % self.checkpoint_every_steps == 0):
                        checkpoint_id = self.global_step // self.checkpoint_every_steps
                        checkpoint_path = os.path.join(
                            self.checkpoint_dir,
                            f'checkpoint_{checkpoint_id}.pt'
                        )
                        self.algo.save(checkpoint_path)
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        except Exception as e:
            print(f"\nTraining interrupted by exception: {e}")
            raise
        finally:
            # Save final checkpoint
            final_checkpoint_path = os.path.join(
                self.checkpoint_dir,
                'checkpoint_final.pt'
            )
            self.algo.save(final_checkpoint_path)
            
            # Close metrics file
            if self.metrics_file is not None:
                self.metrics_file.close()
            
            # Finalize logger
            self.logger.finish()
    
    def _init_metrics_csv(self) -> None:
        """Initialize metrics CSV file."""
        self.metrics_file = open(self.metrics_csv_path, 'w', newline='')
        self.metrics_writer = csv.DictWriter(
            self.metrics_file,
            fieldnames=['step', 'episode', 'episode_return', 'episode_length']
        )
        self.metrics_writer.writeheader()
        self.metrics_file.flush()
    
    def _write_metrics(self, metrics_dict) -> None:
        """Write a row of metrics to CSV."""
        if self.metrics_writer is not None:
            self.metrics_writer.writerow(metrics_dict)
            self.metrics_file.flush()
