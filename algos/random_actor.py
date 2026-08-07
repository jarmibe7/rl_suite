import numpy as np
from typing import Dict
import torch
from algos.base import Algorithm
from gymnasium import spaces
import os


class RandomActor(Algorithm):
    """Random policy that always samples actions uniformly from action space."""
    
    requires_sequences: bool = False
    
    def __init__(self, action_space):
        """Initialize random actor.
        
        Args:
            action_space: Gymnasium action space
        """
        self.action_space = action_space
        self.is_discrete = isinstance(action_space, spaces.Discrete)
        self.action_shape = getattr(action_space, "shape", ())
        if not self.is_discrete:
            self.action_low = torch.tensor(action_space.low, dtype=torch.float32)
            self.action_high = torch.tensor(action_space.high, dtype=torch.float32)
    
    def act(self, obs: np.ndarray, deterministic: bool = False):
        """Sample random action.
        
        Args:
            obs: Observation (unused)
            deterministic: Ignored for random policy
            
        Returns:
            Random action sampled uniformly from action space
        """
        if self.is_discrete:
            return int(self.action_space.sample())
        return self.action_space.sample()
    
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """No-op for random actor.
        
        Args:
            batch: Batch data (unused)
            
        Returns:
            Empty metrics dict
        """
        return {}
    
    def save(self, path: str) -> None:
        """No-op for random actor (stateless).
        
        Args:
            path: Save path (unused)
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"algorithm": "random_actor", "stateless": True}, path)
    
    def load(self, path: str) -> None:
        """No-op for random actor (stateless).
        
        Args:
            path: Load path (unused)
        """
        pass
