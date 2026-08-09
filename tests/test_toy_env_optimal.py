"""Test Gridworld environment and optimal policy."""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.gridworld import Gridworld


class TestGridworldEnvironment:
    """Test Gridworld environment."""
    
    def test_gridworld_reset(self):
        """Test that reset works correctly."""
        env = Gridworld(grid_size=5, max_steps=10)
        obs, info = env.reset(seed=0)
        
        # Check observation shape
        assert obs.shape == (4,), f"Expected shape (4,), got {obs.shape}"
        
        # Check agent starts at (0, 0)
        assert obs[0] == 0 and obs[1] == 0, f"Agent should start at (0, 0), got {obs[:2]}"
        
        # Check goal position in observation
        assert obs[2] == 4 and obs[3] == 4, f"Goal should be at (4, 4), got {obs[2:]}"
    
    def test_gridworld_step(self):
        """Test that step works correctly."""
        env = Gridworld(grid_size=5, max_steps=10)
        obs, _ = env.reset(seed=0)
        
        # Move right
        next_obs, reward, terminated, truncated, info = env.step(0)
        
        # Agent position should change
        assert next_obs[0] == 1 and next_obs[1] == 0, f"After moving right, agent should be at (1, 0), got {next_obs[:2]}"
        
        # Reward should be negative (distance to goal)
        assert reward < 0, f"Reward should be negative, got {reward}"
        
        # Episode should not be done
        assert not terminated and not truncated
    
    def test_gridworld_episode_termination(self):
        """Test that episode terminates when goal is reached."""
        # Create small gridworld where goal is close
        env = Gridworld(grid_size=3, max_steps=10, goal_pos=(1, 0), start_pos=(0, 0))
        obs, _ = env.reset(seed=0)
        
        # Move right (should reach goal)
        next_obs, reward, terminated, truncated, info = env.step(0)
        
        # Episode should be done
        assert terminated, "Episode should terminate when agent reaches goal"
    
    def test_gridworld_max_steps_truncation(self):
        """Test that episode truncates after max steps."""
        env = Gridworld(grid_size=5, max_steps=2)
        obs, _ = env.reset(seed=0)
        
        # Take max_steps steps
        for i in range(2):
            obs, reward, terminated, truncated, info = env.step(0)
            if i == 1:
                # Should be truncated after max_steps
                assert truncated, "Episode should be truncated after max_steps"
    
    def test_gridworld_boundary_conditions(self):
        """Test that agent doesn't move outside grid."""
        env = Gridworld(grid_size=5, max_steps=10, start_pos=(0, 0))
        obs, _ = env.reset(seed=0)
        
        # Try to move left (should be clamped)
        obs, _, _, _, _ = env.step(2)  # 2 = left
        assert obs[0] == 0, "Agent x should be clamped at 0"
        
        # Try to move down from the bottom edge (should be clamped)
        obs, _, _, _, _ = env.step(3)  # 3 = down
        assert obs[1] == 0, f"Agent y should be clamped at 0, got {obs[1]}"
    
    def test_gridworld_optimal_return(self):
        """Test optimal_return calculation."""
        # Simple case: distance 3
        env = Gridworld(grid_size=5, max_steps=10, start_pos=(0, 0), goal_pos=(3, 0))
        optimal = env.optimal_return()
        
        # Optimal path: move right 3 times
        # The environment rewards after each move using the new distance:
        # rewards: -2, -1, 0
        # Total: -3
        expected = -(2 + 1 + 0)
        
        assert optimal == expected, f"Expected optimal return {expected}, got {optimal}"


class OptimalGridworldPolicy:
    """A scripted policy that moves optimally toward goal."""
    
    def __init__(self, env):
        self.env = env
    
    def act(self, obs):
        """
        Move optimally toward goal.
        
        obs is (agent_x, agent_y, goal_x, goal_y)
        """
        agent_x, agent_y, goal_x, goal_y = obs[:4]
        
        # Decide which direction to move
        # Priority: x-direction first, then y-direction
        if agent_x < goal_x:
            return 0  # Move right
        elif agent_x > goal_x:
            return 2  # Move left
        elif agent_y < goal_y:
            return 1  # Move up
        elif agent_y > goal_y:
            return 3  # Move down
        else:
            # Already at goal, can take any action
            return 0


def test_optimal_policy_achieves_optimal_return():
    """Test that optimal policy achieves the computed optimal return."""
    seeds = [0, 1, 2]
    
    for seed in seeds:
        env = Gridworld(grid_size=7, max_steps=20)
        env.reset(seed=seed)
        
        # Compute expected optimal return from the environment's reward convention
        optimal_return = env.optimal_return()
        
        # Run policy
        policy = OptimalGridworldPolicy(env)
        obs, _ = env.reset(seed=seed)
        
        episode_return = 0.0
        done = False
        step_count = 0
        
        while not done and step_count < 100:  # Large safety limit
            action = policy.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += float(reward)
            step_count += 1
        
        # Check that policy achieves near optimal return
        assert abs(episode_return - optimal_return) < 1e-5, (
            f"Policy return {episode_return} does not match "
            f"optimal return {optimal_return} (seed {seed})"
        )
    
    print("Optimal policy test passed!")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
