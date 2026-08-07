from abc import ABC, abstractmethod
from typing import Dict
import numpy as np


class Algorithm(ABC):
    """Base class for all RL algorithms in the suite."""
    
    requires_sequences: bool = False

    @abstractmethod
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select an action given an observation.
        
        Args:
            obs: Current observation
            deterministic: If True, use deterministic policy (e.g., mean of distribution)
            
        Returns:
            Action array
        """
        pass

    @abstractmethod
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Update algorithm parameters with a batch of samples.
        
        Args:
            batch: Dictionary with keys 'obs', 'action', 'reward', 'next_obs', 'done'
                   or sequences of these depending on requires_sequences
                   
        Returns:
            Dictionary of metrics to log
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save algorithm state to disk.
        
        Args:
            path: Path to save to
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load algorithm state from disk.
        
        Args:
            path: Path to load from
        """
        pass

    def reset(self) -> None:
        """Reset any internal state.
        
        Default no-op; override for algorithms with recurrent state.
        """
        pass
