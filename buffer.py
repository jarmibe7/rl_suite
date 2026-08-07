"""Replay buffer for storing and sampling transitions or sequences."""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque


class ReplayBuffer:
    """Ring buffer for storing transitions or sequences.
    
    When sequence_length is None: transition mode
        - add_step(obs, action, reward, next_obs, done)
        - sample(batch_size) returns flat {obs, action, reward, next_obs, done}
    
    When sequence_length is specified: sequence mode
        - add_episode(episode_dict) where episode_dict has keys obs, action, reward, done
        - sample(batch_size) returns (T=sequence_length, B=batch_size, ...) chunks
    """
    
    def __init__(
        self,
        capacity: int,
        sequence_length: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions (transition mode) or episodes (sequence mode)
            sequence_length: If None, use transition mode. Otherwise sequence mode with this length.
            dtype: PyTorch dtype for storing data
        """
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.dtype = dtype
        self._sequence_mode = sequence_length is not None
        
        if self._sequence_mode:
            # Sequence mode: store full episodes
            self.episodes = deque(maxlen=capacity)
            self.num_episodes = 0
        else:
            # Transition mode: store (obs, action, reward, next_obs, done) tuples
            self.observations = deque(maxlen=capacity)
            self.actions = deque(maxlen=capacity)
            self.rewards = deque(maxlen=capacity)
            self.next_observations = deque(maxlen=capacity)
            self.dones = deque(maxlen=capacity)
            self.num_transitions = 0
    
    @property
    def sequence_mode(self) -> bool:
        """Whether buffer is in sequence mode."""
        return self._sequence_mode
    
    def __len__(self) -> int:
        """Return number of stored items (transitions or episodes)."""
        if self._sequence_mode:
            return len(self.episodes)
        else:
            return len(self.observations)
    
    def ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch.
        
        Args:
            batch_size: Minimum batch size needed
            
        Returns:
            True if buffer has at least batch_size samples
        """
        return len(self) >= batch_size
    
    def add_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add a single transition (transition mode only).
        
        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode ended
        """
        if self._sequence_mode:
            raise RuntimeError("Cannot use add_step in sequence mode. Use add_episode instead.")
        
        self.observations.append(torch.tensor(obs, dtype=self.dtype))
        self.actions.append(torch.tensor(action, dtype=self.dtype))
        self.rewards.append(torch.tensor(reward, dtype=self.dtype))
        self.next_observations.append(torch.tensor(next_obs, dtype=self.dtype))
        self.dones.append(torch.tensor(done, dtype=torch.bool))
        self.num_transitions += 1
    
    def add_episode(self, episode_dict: Dict[str, np.ndarray]) -> None:
        """Add a full episode (sequence mode only).
        
        Args:
            episode_dict: Dictionary with keys 'obs', 'action', 'reward', 'done'
                         Each value should be an array of shape (T, ...)
        """
        if not self._sequence_mode:
            raise RuntimeError("Cannot use add_episode in transition mode. Use add_step instead.")
        
        required_keys = {'obs', 'action', 'reward', 'done'}
        if not required_keys.issubset(episode_dict.keys()):
            raise ValueError(f"episode_dict must contain keys {required_keys}")
        
        # Convert to tensors and store
        episode = {
            'obs': torch.tensor(episode_dict['obs'], dtype=self.dtype),
            'action': torch.tensor(episode_dict['action'], dtype=self.dtype),
            'reward': torch.tensor(episode_dict['reward'], dtype=self.dtype),
            'done': torch.tensor(episode_dict['done'], dtype=torch.bool),
        }
        
        self.episodes.append(episode)
        self.num_episodes += 1
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions or sequences.
        
        Args:
            batch_size: Batch size
            
        Returns:
            In transition mode: {obs, action, reward, next_obs, done} with shape (B, ...)
            In sequence mode: {obs, action, reward, done} with shape (T, B, ...)
        """
        if not self.ready(batch_size):
            raise RuntimeError(
                f"Buffer not ready. Need {batch_size} samples, have {len(self)}."
            )
        
        if self._sequence_mode:
            return self._sample_sequences(batch_size)
        else:
            return self._sample_transitions(batch_size)
    
    def _sample_transitions(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions (transition mode)."""
        indices = np.random.choice(len(self.observations), size=batch_size, replace=False)
        
        batch = {
            'obs': torch.stack([self.observations[i] for i in indices]),
            'action': torch.stack([self.actions[i] for i in indices]),
            'reward': torch.stack([self.rewards[i] for i in indices]),
            'next_obs': torch.stack([self.next_observations[i] for i in indices]),
            'done': torch.stack([self.dones[i] for i in indices]),
        }
        
        return batch
    
    def _sample_sequences(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of sequences (sequence mode)."""
        episodes_list = list(self.episodes)
        indices = np.random.choice(len(episodes_list), size=batch_size, replace=False)
        
        # Extract sequences of length sequence_length from episodes
        sequences = {
            'obs': [],
            'action': [],
            'reward': [],
            'done': [],
        }
        
        for idx in indices:
            episode = episodes_list[idx]
            episode_len = len(episode['obs'])
            
            # Sample random start position in episode
            if episode_len >= self.sequence_length:
                start = np.random.randint(0, episode_len - self.sequence_length + 1)
            else:
                # Episode shorter than sequence_length; pad with zeros or use as is
                start = 0
            
            end = min(start + self.sequence_length, episode_len)
            seq_len = end - start
            
            # Extract sequence
            for key in sequences.keys():
                seq = episode[key][start:end]
                
                # Pad if necessary
                if seq_len < self.sequence_length:
                    pad_len = self.sequence_length - seq_len
                    padding = torch.zeros(
                        (pad_len,) + seq.shape[1:],
                        dtype=seq.dtype,
                        device=seq.device,
                    )
                    seq = torch.cat([seq, padding], dim=0)
                
                sequences[key].append(seq)
        
        # Stack into batches: (T, B, ...)
        batch = {
            key: torch.stack(sequences[key], dim=1)
            for key in sequences.keys()
        }
        
        return batch
