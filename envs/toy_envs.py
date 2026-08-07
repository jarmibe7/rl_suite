"""Gridworld environment - deterministic sanity check environment."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Gridworld(gym.Env):
    """Simple gridworld where an agent moves on a grid to reach a goal.
    
    State: 2D position (x, y)
    Actions: 4 discrete directions (up, down, left, right)
    Reward: -1 * manhattan_distance_to_goal (dense reward)
    Episode ends when agent reaches goal or max_steps exceeded.
    
    Purpose: Any correctly wired RL algorithm should achieve near-optimal
    return on this simple environment.
    """
    
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 4}
    
    def __init__(
        self,
        grid_size: int = 5,
        max_steps: int = 10,
        goal_pos: tuple = None,
        start_pos: tuple = None,
        size_obs: bool = False,
    ):
        """Initialize Gridworld.
        
        Args:
            grid_size: Size of square grid (grid_size x grid_size)
            max_steps: Maximum steps per episode
            goal_pos: (x, y) goal position. If None, placed at (grid_size-1, grid_size-1)
            start_pos: (x, y) start position. If None, placed at (0, 0)
            size_obs: If True, return observation as (grid_size, grid_size) grid with agent
                      marked as 1 and goal as 2. If False, return (x, y, goal_x, goal_y).
        """
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.size_obs = size_obs
        
        # Set positions
        self.goal_pos = goal_pos if goal_pos is not None else (grid_size - 1, grid_size - 1)
        self.start_pos = start_pos if start_pos is not None else (0, 0)
        
        # Agent position
        self.agent_pos = None
        self.step_count = 0
        
        # Action space: 4 discrete directions (0=right, 1=up, 2=left, 3=down)
        self.action_space = spaces.Discrete(4)
        
        # Observation space
        if size_obs:
            # Grid representation
            self.observation_space = spaces.Box(
                low=0,
                high=2,
                shape=(grid_size, grid_size),
                dtype=np.float32,
            )
        else:
            # (agent_x, agent_y, goal_x, goal_y)
            self.observation_space = spaces.Box(
                low=0,
                high=grid_size - 1,
                shape=(4,),
                dtype=np.float32,
            )
        
        self.render_mode = 'rgb_array'
    
    def reset(self, seed: int = None, options: dict = None):
        """Reset environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Observation and info dict
        """
        super().reset(seed=seed)
        
        self.agent_pos = np.array(self.start_pos, dtype=np.float32)
        self.step_count = 0
        
        return self._get_obs(), {}
    
    def step(self, action: int):
        """Step environment.
        
        Args:
            action: Action (0=right, 1=up, 2=left, 3=down)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Apply action to move agent
        # 0=right, 1=up, 2=left, 3=down
        dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][action]
        
        new_pos = self.agent_pos + np.array([dx, dy], dtype=np.float32)
        
        # Clamp to grid boundaries
        new_pos[0] = np.clip(new_pos[0], 0, self.grid_size - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.grid_size - 1)
        
        self.agent_pos = new_pos
        self.step_count += 1
        
        # Compute reward: negative manhattan distance to goal
        obs = self._get_obs()
        reward = -self._manhattan_distance(self.agent_pos, np.array(self.goal_pos))
        
        # Check if goal reached
        terminated = np.allclose(self.agent_pos, self.goal_pos)
        truncated = self.step_count >= self.max_steps
        
        info = {'agent_pos': tuple(self.agent_pos), 'goal_pos': self.goal_pos}
        
        return obs, float(reward), terminated, truncated, info
    
    def render(self):
        """Render environment as RGB image.
        
        Returns:
            RGB array of shape (grid_size*32, grid_size*32, 3)
        """
        cell_size = 32
        img = np.ones((self.grid_size * cell_size, self.grid_size * cell_size, 3), dtype=np.uint8) * 255
        
        # Draw goal (green)
        goal_x, goal_y = int(self.goal_pos[0]), int(self.goal_pos[1])
        img[
            goal_y * cell_size:(goal_y + 1) * cell_size,
            goal_x * cell_size:(goal_x + 1) * cell_size
        ] = [0, 255, 0]
        
        # Draw agent (red)
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        img[
            agent_y * cell_size:(agent_y + 1) * cell_size,
            agent_x * cell_size:(agent_x + 1) * cell_size
        ] = [255, 0, 0]
        
        return img
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        if self.size_obs:
            # Grid representation
            obs = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
            goal_x, goal_y = int(self.goal_pos[0]), int(self.goal_pos[1])
            obs[agent_y, agent_x] = 1.0
            obs[goal_y, goal_x] = 2.0
            return obs
        else:
            # Coordinate representation (agent_x, agent_y, goal_x, goal_y)
            return np.concatenate([
                self.agent_pos,
                np.array(self.goal_pos, dtype=np.float32)
            ])
    
    def _manhattan_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Compute manhattan distance between two positions."""
        return float(np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1]))
    
    def optimal_return(self) -> float:
        """Compute optimal return for this episode setup.
        
        Assuming dense reward of -manhattan_distance_to_goal at each step,
        the optimal policy moves directly toward the goal and achieves:
        sum of -distance_at_each_step.
        
        Starting at start_pos, moving optimally to goal_pos, then staying at goal.
        
        Returns:
            Best possible cumulative reward
        """
        dist_to_goal = self._manhattan_distance(
            np.array(self.start_pos, dtype=np.float32),
            np.array(self.goal_pos, dtype=np.float32)
        )
        
        # Optimal path takes exactly dist_to_goal steps to reach goal
        # After reaching goal, algorithm is done (no further steps)
        optimal_return = 0.0
        
        # At each step moving toward goal, distance decreases by 1
        for step in range(int(dist_to_goal)):
            remaining_dist = dist_to_goal - step
            optimal_return -= remaining_dist
        
        return optimal_return
