"""Tests for ReplayBuffer."""

import pytest
import numpy as np
import torch
from buffer import ReplayBuffer


class TestReplayBufferTransitionMode:
    """Test ReplayBuffer in transition mode (sequence_length=None)."""
    
    def test_add_step_and_sample(self):
        """Test adding steps and sampling transitions."""
        buffer = ReplayBuffer(capacity=100, sequence_length=None)
        
        # Add some steps
        for i in range(10):
            obs = np.array([i, i+1], dtype=np.float32)
            action = np.array([i*0.1], dtype=np.float32)
            reward = float(i)
            next_obs = np.array([i+1, i+2], dtype=np.float32)
            done = (i % 3 == 0)
            
            buffer.add_step(obs, action, reward, next_obs, done)
        
        # Check buffer state
        assert len(buffer) == 10
        assert buffer.ready(5)
        assert not buffer.ready(20)
        assert not buffer.sequence_mode
    
    def test_sample_shapes(self):
        """Test that sampled batch has correct shapes."""
        buffer = ReplayBuffer(capacity=100, sequence_length=None)
        
        # Add steps with specific shapes
        obs_shape = (4,)
        action_shape = (2,)
        
        for i in range(20):
            obs = np.random.randn(*obs_shape).astype(np.float32)
            action = np.random.randn(*action_shape).astype(np.float32)
            reward = np.random.randn()
            next_obs = np.random.randn(*obs_shape).astype(np.float32)
            done = np.random.rand() > 0.8
            
            buffer.add_step(obs, action, reward, next_obs, done)
        
        # Sample batch
        batch_size = 5
        batch = buffer.sample(batch_size)
        
        # Check shapes
        assert batch['obs'].shape == (batch_size, *obs_shape)
        assert batch['action'].shape == (batch_size, *action_shape)
        assert batch['reward'].shape == (batch_size,)
        assert batch['next_obs'].shape == (batch_size, *obs_shape)
        assert batch['done'].shape == (batch_size,)
        
        # Check types
        assert isinstance(batch['obs'], torch.Tensor)
        assert batch['obs'].dtype == torch.float32
    
    def test_ring_buffer_behavior(self):
        """Test that buffer correctly overwrites old data when full."""
        buffer = ReplayBuffer(capacity=10, sequence_length=None)
        
        # Add more than capacity
        for i in range(20):
            obs = np.array([i], dtype=np.float32)
            action = np.array([0], dtype=np.float32)
            buffer.add_step(obs, action, 0.0, obs, False)
        
        # Buffer should only contain last 10 items
        assert len(buffer) == 10


class TestReplayBufferSequenceMode:
    """Test ReplayBuffer in sequence mode (sequence_length specified)."""
    
    def test_add_episode_and_sample(self):
        """Test adding episodes and sampling sequences."""
        seq_len = 5
        buffer = ReplayBuffer(capacity=10, sequence_length=seq_len)
        
        assert buffer.sequence_mode
        assert not buffer.ready(1)
        
        # Add episode
        episode = {
            'obs': np.random.randn(10, 4).astype(np.float32),
            'action': np.random.randn(10, 2).astype(np.float32),
            'reward': np.random.randn(10).astype(np.float32),
            'done': np.random.rand(10) > 0.8,
        }
        
        buffer.add_episode(episode)
        assert len(buffer) == 1
        assert buffer.ready(1)
    
    def test_sample_sequence_shapes(self):
        """Test that sampled sequences have correct shapes."""
        seq_len = 5
        buffer = ReplayBuffer(capacity=10, sequence_length=seq_len)
        
        # Add multiple episodes
        for ep in range(3):
            episode = {
                'obs': np.random.randn(10, 4).astype(np.float32),
                'action': np.random.randn(10, 2).astype(np.float32),
                'reward': np.random.randn(10).astype(np.float32),
                'done': np.random.rand(10) > 0.8,
            }
            buffer.add_episode(episode)
        
        # Sample batch
        batch_size = 2
        batch = buffer.sample(batch_size)
        
        # Check shapes: (T, B, ...)
        assert batch['obs'].shape == (seq_len, batch_size, 4)
        assert batch['action'].shape == (seq_len, batch_size, 2)
        assert batch['reward'].shape == (seq_len, batch_size)
        assert batch['done'].shape == (seq_len, batch_size)
        
        # Check types
        assert isinstance(batch['obs'], torch.Tensor)
        assert batch['obs'].dtype == torch.float32
    
    def test_short_episode_padding(self):
        """Test that episodes shorter than seq_len are handled."""
        seq_len = 10
        buffer = ReplayBuffer(capacity=10, sequence_length=seq_len)
        
        # Add short episode
        episode = {
            'obs': np.random.randn(5, 4).astype(np.float32),  # Shorter than seq_len
            'action': np.random.randn(5, 2).astype(np.float32),
            'reward': np.random.randn(5).astype(np.float32),
            'done': np.ones(5, dtype=bool),
        }
        
        buffer.add_episode(episode)
        
        # Should still sample correctly (with padding)
        batch = buffer.sample(1)
        assert batch['obs'].shape == (seq_len, 1, 4)


class TestBufferModeValidation:
    """Test that buffer validates mode correctly."""
    
    def test_add_step_in_sequence_mode_raises(self):
        """Test that add_step raises error in sequence mode."""
        buffer = ReplayBuffer(capacity=100, sequence_length=5)
        
        with pytest.raises(RuntimeError):
            buffer.add_step(
                np.array([1.0, 2.0]),
                np.array([0.1]),
                1.0,
                np.array([1.1, 2.1]),
                False
            )
    
    def test_add_episode_in_transition_mode_raises(self):
        """Test that add_episode raises error in transition mode."""
        buffer = ReplayBuffer(capacity=100, sequence_length=None)
        
        episode = {
            'obs': np.random.randn(5, 4),
            'action': np.random.randn(5, 2),
            'reward': np.random.randn(5),
            'done': np.zeros(5, dtype=bool),
        }
        
        with pytest.raises(RuntimeError):
            buffer.add_episode(episode)
    
    def test_sample_not_ready_raises(self):
        """Test that sampling empty buffer raises error."""
        buffer = ReplayBuffer(capacity=100, sequence_length=None)
        
        with pytest.raises(RuntimeError):
            buffer.sample(10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
