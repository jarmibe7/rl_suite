"""Environment wrappers to standardize observation and action interfaces."""

import gymnasium as gym
import numpy as np
from typing import Optional, Any


GYM_ENV_ALIASES = {
    'Pendulum': 'Pendulum-v1',
    'Pendulum-v1': 'Pendulum-v1',
    'MountainCarContinuous': 'MountainCarContinuous-v0',
    'MountainCarContinuous-v0': 'MountainCarContinuous-v0',
    'HalfCheetah': 'HalfCheetah-v5',
    'HalfCheetah-v5': 'HalfCheetah-v5',
    'Humanoid': 'Humanoid-v5',
    'Humanoid-v5': 'Humanoid-v5'
}

# Env ID prefixes that use MuJoCo's OpenGL offscreen renderer
MUJOCO_ENV_PREFIXES = (
    'HalfCheetah', 'Hopper', 'Walker2d', 'Ant', 'Humanoid',
    'Swimmer', 'Reacher', 'InvertedPendulum', 'InvertedDoublePendulum', 'Pusher',
)

_virtual_display = None


def _ensure_virtual_display() -> None:
    """Start a virtual X display (once per process) so MuJoCo can create an
    OpenGL context for offscreen rendering when there is no real display (e.g. over SSH).
    """
    global _virtual_display
    if _virtual_display is None:
        from pyvirtualdisplay import Display
        _virtual_display = Display(visible=False, size=(1024, 768))
        _virtual_display.start()


class ObsActionWrapper(gym.Wrapper):
    """Base wrapper for standardizing observation and action interfaces.
    
    Ensures observations and actions are in consistent format (numpy arrays).
    """
    
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        """Step environment with standardized action."""
        # Convert action to numpy array if needed
        if isinstance(action, (list, tuple)):
            action = np.array(action, dtype=np.float32)
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Ensure observation is numpy array
        obs = self._standardize_obs(obs)
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment with standardized observation."""
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Ensure observation is numpy array
        obs = self._standardize_obs(obs)
        
        return obs, info
    
    def _standardize_obs(self, obs):
        """Convert observation to numpy array."""
        if isinstance(obs, dict):
            # If observation is a dict, we could flatten or return as is
            # For now, raise an error to handle explicitly if needed
            raise NotImplementedError(
                f"Dict observations not automatically handled. "
                f"Implement custom wrapper. Keys: {obs.keys()}"
            )
        
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
        else:
            obs = obs.astype(np.float32)
        
        return obs


class FlattenObsWrapper(gym.Wrapper):
    """Flatten multi-dimensional observations to 1D vectors."""
    
    def __init__(self, env):
        super().__init__(env)
        
        # Compute flat observation space
        if isinstance(env.observation_space, gym.spaces.Box):
            self.flat_dim = int(np.prod(env.observation_space.shape))
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.flat_dim,),
                dtype=np.float32,
            )
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = obs.flatten().astype(np.float32)
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = obs.flatten().astype(np.float32)
        return obs, info


class NormObsWrapper(gym.Wrapper):
    """Normalize observations to [-1, 1] range based on space bounds.
    
    Only works with Box observation spaces.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError("NormObsWrapper only supports Box observation spaces")
        
        self.obs_low = env.observation_space.low.astype(np.float32)
        self.obs_high = env.observation_space.high.astype(np.float32)
        
        # Avoid division by zero
        range_size = self.obs_high - self.obs_low
        range_size[range_size == 0] = 1.0
        self.range_size = range_size
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._normalize_obs(obs)
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._normalize_obs(obs)
        return obs, info
    
    def _normalize_obs(self, obs):
        """Normalize observation to [-1, 1]."""
        obs = obs.astype(np.float32)
        # Shift to [0, 1]
        obs = (obs - self.obs_low) / self.range_size
        # Shift to [-1, 1]
        obs = 2 * obs - 1
        # Clip to [-1, 1]
        obs = np.clip(obs, -1, 1)
        return obs


def make_env(
    env_id: str,
    seed: int = 0,
    render_mode: Optional[str] = None,
    **kwargs,
) -> gym.Env:
    """Create and wrap a Gymnasium environment.
    
    Args:
        env_id: Environment ID (e.g., 'CartPole-v1', 'Gridworld', 'PointGoal')
        seed: Random seed
        render_mode: Optional Gymnasium render mode
        **kwargs: Additional arguments for environment
        
    Returns:
        Wrapped environment
    """
    env_id = GYM_ENV_ALIASES.get(env_id, env_id)

    # Special handling for custom environments
    if env_id == 'Gridworld':
        from envs.gridworld import Gridworld
        env = Gridworld(**kwargs)
    elif env_id == 'PointGoal':
        from envs.gridworld import PointGoal
        env = PointGoal(**kwargs)
    else:
        # MuJoCo envs need a virtual display to render over ssh
        if render_mode is not None and env_id.startswith(MUJOCO_ENV_PREFIXES):
            _ensure_virtual_display()

        # Standard Gymnasium environments
        if render_mode is None:
            env = gym.make(env_id)
        else:
            env = gym.make(env_id, render_mode=render_mode)

    # Wrap with standardization
    env = ObsActionWrapper(env)
    
    # Set seed
    env.reset(seed=seed)
    
    return env
