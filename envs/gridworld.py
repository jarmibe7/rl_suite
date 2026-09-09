"""Gridworld environment - test env for all algorithms."""

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
        random_start: bool = False,
        random_goal: bool = False,
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
        self.random_start = random_start
        self.random_goal = random_goal
        
        # Store the configured positions as the deterministic defaults
        self._default_goal_pos = goal_pos if goal_pos is not None else (grid_size - 1, grid_size - 1)
        self._default_start_pos = start_pos if start_pos is not None else (0, 0)
        self.goal_pos = self._default_goal_pos
        self.start_pos = self._default_start_pos
        
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

        if self.random_start or self.random_goal:
            start_pos, goal_pos = self._sample_positions()
            self.start_pos = start_pos
            self.goal_pos = goal_pos
        else:
            self.start_pos = self._default_start_pos
            self.goal_pos = self._default_goal_pos
        
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
        reward = -self._manhattan_distance(self.agent_pos, np.array(self.goal_pos)) / self.grid_size
        
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
    
    def _sample_positions(self) -> tuple:
        """Sample a new start/goal pair, ensuring they are distinct when both are random."""
        if self.random_start and self.random_goal:
            while True:
                start = tuple(self._np_random.integers(0, self.grid_size, size=2))
                goal = tuple(self._np_random.integers(0, self.grid_size, size=2))
                if start != goal:
                    return start, goal

        if self.random_start:
            start = tuple(self._np_random.integers(0, self.grid_size, size=2))
            return start, self._default_goal_pos

        if self.random_goal:
            goal = tuple(self._np_random.integers(0, self.grid_size, size=2))
            return self._default_start_pos, goal

        return self._default_start_pos, self._default_goal_pos

    def _manhattan_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Compute manhattan distance between two positions."""
        return float(np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1]))
    
    def optimal_return(self) -> float:
        """Compute optimal return for this episode setup.
        
        Rewards are emitted after a move based on the new distance to the goal.
        For an optimal path, the rewards are:
        -(d-1), -(d-2), ..., -1, 0
        where d is the initial Manhattan distance to the goal.
        
        Returns:
            Best possible cumulative reward
        """
        dist_to_goal = self._manhattan_distance(
            np.array(self.start_pos, dtype=np.float32),
            np.array(self.goal_pos, dtype=np.float32)
        )

        if dist_to_goal <= 0:
            return 0.0

        optimal_return = 0.0
        for step in range(int(dist_to_goal) - 1, -1, -1):
            optimal_return -= step

        return optimal_return / self.grid_size


class PointGoal(gym.Env):
    """Continuous point navigation environment.
    
    An agent controls the acceleration of a point on a 2D plane to reach a goal.
    
    State: 2D position (x, y) and velocity (vx, vy)
    Actions: 2D acceleration (ax, ay)
    Reward: -euclidean_distance_to_goal (dense reward)
    Episode ends when agent reaches goal or max_steps exceeded.
    
    Purpose: Continuous-control version of Gridworld for testing continuous RL algs
    """
    
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 4}
    
    def __init__(
        self,
        plane_size: int = 10,
        max_steps: int = 100,
        max_speed: float = 1.0,
        goal_threshold: float = 0.1,
        goal_pos: tuple = None,
        start_pos: tuple = None,
        random_start: bool = False,
        random_goal: bool = False,
    ):
        """Initialize PointGoal.
        
        Args:
            plane_size: Size of square plane (plane spans [-plane_size/2, plane_size/2])
            max_steps: Maximum steps per episode
            max_speed: Maximum velocity magnitude per dimension
            goal_threshold: Distance threshold to consider goal reached
            goal_pos: (x, y) goal position. If None, placed at (plane_size/2 - 1, plane_size/2 - 1)
            start_pos: (x, y) start position. If None, placed at (-plane_size/2 + 1, -plane_size/2 + 1)
            random_start: If True, randomly sample start position
            random_goal: If True, randomly sample goal position
        """
        self.plane_size = plane_size
        self.max_steps = max_steps
        self.max_speed = max_speed
        self.goal_threshold = goal_threshold
        self.random_start = random_start
        self.random_goal = random_goal
        
        half_size = plane_size / 2.0
        
        # Store the configured positions as the deterministic defaults
        if goal_pos is None:
            goal_pos = (half_size - 1.0, half_size - 1.0)
        if start_pos is None:
            start_pos = (-half_size + 1.0, -half_size + 1.0)
        
        self._default_goal_pos = goal_pos
        self._default_start_pos = start_pos
        self.goal_pos = self._default_goal_pos
        self.start_pos = self._default_start_pos
        
        # Agent position and velocity
        self.agent_pos = None
        self.agent_vel = None
        self.step_count = 0
        
        # Action space: 2D continuous acceleration
        self.action_space = spaces.Box(
            low=-max_speed,
            high=max_speed,
            shape=(2,),
            dtype=np.float32,
        )
        
        # Observation space: (agent_x, agent_y, goal_x, goal_y)
        self.observation_space = spaces.Box(
            low=-half_size,
            high=half_size,
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

        if self.random_start or self.random_goal:
            start_pos, goal_pos = self._sample_positions()
            self.start_pos = start_pos
            self.goal_pos = goal_pos
        else:
            self.start_pos = self._default_start_pos
            self.goal_pos = self._default_goal_pos
        
        self.agent_pos = np.array(self.start_pos, dtype=np.float32)
        self.agent_vel = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        
        return self._get_obs(), {}
    
    def step(self, action: np.ndarray):
        """Step environment.
        
        Args:
            action: 2D acceleration action
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Ensure action is numpy array
        action = np.array(action, dtype=np.float32)
        
        # Update velocity by acceleration
        self.agent_vel = self.agent_vel + action
        
        # Clip velocity to max_speed
        self.agent_vel = np.clip(self.agent_vel, -self.max_speed, self.max_speed)
        
        # Update position via velocity
        self.agent_pos = self.agent_pos + self.agent_vel
        
        # Clamp position to plane boundaries
        half_size = self.plane_size / 2.0
        self.agent_pos[0] = np.clip(self.agent_pos[0], -half_size, half_size)
        self.agent_pos[1] = np.clip(self.agent_pos[1], -half_size, half_size)
        
        self.step_count += 1
        
        # Compute reward: negative euclidean distance to goal
        obs = self._get_obs()
        reward = -self._euclidean_distance(self.agent_pos, np.array(self.goal_pos)) / self.plane_size
        # reward *= 0.01
        
        # Check if goal reached
        dist = self._euclidean_distance(self.agent_pos, np.array(self.goal_pos))
        terminated = dist < self.goal_threshold
        truncated = self.step_count >= self.max_steps
        
        info = {'agent_pos': tuple(self.agent_pos), 'goal_pos': self.goal_pos}
        
        return obs, float(reward), terminated, truncated, info
    
    def render(self):
        """Render environment as RGB image.
        
        Returns:
            RGB array of shape (512, 512, 3)
        """
        size = 512
        half_size = self.plane_size / 2.0
        
        img = np.ones((size, size, 3), dtype=np.uint8) * 255
        
        # Convert continuous coordinates to pixel coordinates
        def to_pixel(pos):
            x = int((pos[0] + half_size) / self.plane_size * size)
            y = int((pos[1] + half_size) / self.plane_size * size)
            return np.clip(x, 0, size - 1), np.clip(y, 0, size - 1)
        
        # Draw goal (green)
        goal_x, goal_y = to_pixel(self.goal_pos)
        radius = int(self.goal_threshold / self.plane_size * size)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    px, py = goal_x + dx, goal_y + dy
                    if 0 <= px < size and 0 <= py < size:
                        img[py, px] = [0, 255, 0]
        
        # Draw agent (red)
        agent_x, agent_y = to_pixel(self.agent_pos)
        radius = 6
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    px, py = agent_x + dx, agent_y + dy
                    if 0 <= px < size and 0 <= py < size:
                        img[py, px] = [255, 0, 0]
        
        return img
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        return np.concatenate([
            self.agent_pos,
            np.array(self.goal_pos, dtype=np.float32)
        ])
    
    def _sample_positions(self) -> tuple:
        """Sample a new start/goal pair, ensuring they are distinct when both are random."""
        half_size = self.plane_size / 2.0
        low = -half_size
        high = half_size
        
        if self.random_start and self.random_goal:
            while True:
                start = tuple(self._np_random.uniform(low, high, size=2))
                goal = tuple(self._np_random.uniform(low, high, size=2))
                if self._euclidean_distance(np.array(start), np.array(goal)) > 1.0:
                    return start, goal

        if self.random_start:
            start = tuple(self._np_random.uniform(low, high, size=2))
            return start, self._default_goal_pos

        if self.random_goal:
            goal = tuple(self._np_random.uniform(low, high, size=2))
            return self._default_start_pos, goal

        return self._default_start_pos, self._default_goal_pos
    
    def _euclidean_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Compute euclidean distance between two positions."""
        return float(np.linalg.norm(pos1 - pos2))

    # TODO: Test this
    def optimal_return(self) -> float:
        """Compute optimal return for this episode setup.
        
        Assumes straight-line optimal path to goal.
        For continuous control, we approximate by assuming agent can travel
        at max_speed each step.
        
        Returns:
            Best possible cumulative reward (approximate)
        """
        dist_to_goal = self._euclidean_distance(
            np.array(self.start_pos, dtype=np.float32),
            np.array(self.goal_pos, dtype=np.float32)
        )
        
        # Minimum steps needed at max_speed (with goal_threshold buffer)
        min_steps = max(1, int(np.ceil((dist_to_goal - self.goal_threshold) / self.max_speed)))
        
        # Approximate optimal return: accumulate rewards along optimal path
        # This is a rough approximation
        return -dist_to_goal  # Simple lower bound
