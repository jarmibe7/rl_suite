"""Small single-player Pong environment for discrete-control experiments."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Pong(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 30}

    def __init__(self, max_steps: int = 500, render_mode: str = None):
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)  # stay, up, down
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1, -1, 0, 0], dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )
        self._width = 320
        self._height = 240
        self._paddle_speed = 0.04
        self._ball_speed = 0.025
        self._paddle_height = 0.2
        self._player_x = 0.08
        self._opponent_x = 0.92
        self.reset()

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        self.ball = np.array([0.5, 0.5], dtype=np.float32)
        self.velocity = np.array([
            self._ball_speed * self.np_random.choice([-1, 1]),
            self.np_random.uniform(-self._ball_speed / 2, self._ball_speed / 2),
        ], dtype=np.float32)
        self.player_y = 0.5
        self.opponent_y = 0.5
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action: int):
        if action == 1:
            self.player_y -= self._paddle_speed
        elif action == 2:
            self.player_y += self._paddle_speed
        paddle_y_min = self._paddle_height / 2
        paddle_y_max = 1 - paddle_y_min
        self.player_y = float(np.clip(self.player_y, paddle_y_min, paddle_y_max))

        opponent_delta = self.ball[1] - self.opponent_y
        opponent_deadzone = self._paddle_height / 2
        if self.velocity[0] > 0 and abs(opponent_delta) > opponent_deadzone:
            opponent_step = np.sign(opponent_delta) * min(
                self._paddle_speed * 0.75,
                abs(opponent_delta) - opponent_deadzone,
            )
            self.opponent_y = float(np.clip(
                self.opponent_y + opponent_step,
                paddle_y_min,
                paddle_y_max,
            ))

        previous_ball_x = self.ball[0]
        self.ball += self.velocity
        if self.ball[1] <= 0 or self.ball[1] >= 1:
            self.ball[1] = np.clip(self.ball[1], 0, 1)
            self.velocity[1] *= -1

        reward = 0.0
        terminated = False
        player_hit_x = self._player_x + 0.025
        opponent_hit_x = self._opponent_x - 0.025
        if (
            self.velocity[0] < 0
            and previous_ball_x > player_hit_x
            and self.ball[0] <= player_hit_x
        ):
            hit_half_height = self._paddle_height / 2 + 4 / self._height
            if abs(self.ball[1] - self.player_y) <= hit_half_height:
                self.ball[0] = player_hit_x
                self.velocity[0] = abs(self.velocity[0])
                self.velocity[1] += (self.ball[1] - self.player_y) * 0.02
        elif (
            self.velocity[0] > 0
            and previous_ball_x < opponent_hit_x
            and self.ball[0] >= opponent_hit_x
        ):
            hit_half_height = self._paddle_height / 2 + 4 / self._height
            if abs(self.ball[1] - self.opponent_y) <= hit_half_height:
                self.ball[0] = opponent_hit_x
                self.velocity[0] = -abs(self.velocity[0])
                self.velocity[1] += (self.ball[1] - self.opponent_y) * 0.02

        if self.ball[0] <= -4 / self._width:
            reward = -1.0
            terminated = True
        elif self.ball[0] >= 1 + 4 / self._width:
            reward = 1.0
            terminated = True

        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != 'rgb_array':
            return None
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        center_x = self._width // 2
        frame[:, center_x - 1:center_x + 1] = 80
        self._draw_paddle(frame, self._player_x, self.player_y)
        self._draw_paddle(frame, self._opponent_x, self.opponent_y)
        ball_x, ball_y = (self.ball * [self._width, self._height]).astype(int)
        top = max(0, ball_y - 4)
        bottom = min(self._height, ball_y + 4)
        left = max(0, ball_x - 4)
        right = min(self._width, ball_x + 4)
        frame[top:bottom, left:right] = 255
        return frame

    def _get_obs(self):
        return np.array([
            self.ball[0], self.ball[1],
            np.clip(self.velocity[0] / (self._ball_speed * 2), -1, 1),
            np.clip(self.velocity[1] / (self._ball_speed * 2), -1, 1),
            self.player_y, self.opponent_y,
        ], dtype=np.float32)

    def _draw_paddle(self, frame, x, y):
        center_x = int(x * self._width)
        center_y = int(y * self._height)
        half_height = int(self._paddle_height * self._height / 2)
        frame[center_y - half_height:center_y + half_height, center_x - 3:center_x + 3] = 255