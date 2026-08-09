"""Compat tests for continuous classic-control environments."""

import sys
from pathlib import Path
import warnings

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.wrappers import make_env


def _exercise_env(env_name: str, expected_obs_shape: tuple[int, ...]) -> None:
    env = make_env(env_name, seed=0, render_mode='rgb_array')
    obs, info = env.reset(seed=0)

    assert obs.shape == expected_obs_shape
    assert obs.dtype == np.float32

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.shape == expected_obs_shape
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    # Test rendering
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        frame = env.render()

    assert frame is not None
    assert len(caught) == 0


def test_pendulum_alias_support():
    _exercise_env('Pendulum', (3,))


def test_mountain_car_continuous_alias_support():
    _exercise_env('MountainCarContinuous', (2,))