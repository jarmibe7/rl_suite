"""Algorithm-agnostic evaluation utilities."""

import numpy as np
import os
import subprocess
import shutil
from typing import Dict, Optional


class Evaluator:
    """Evaluate an algorithm on an environment without updating weights.
    
    Runs deterministic rollouts and optionally records videos.
    """
    
    def __init__(
        self,
        env,
        algo,
        run_dir: str,
        num_episodes: int = 5,
        record_video: bool = True,
        video_every_calls: int = 1,
    ):
        """Initialize evaluator.
        
        Args:
            env: Gymnasium environment
            algo: Algorithm instance (will call act(obs, deterministic=True) and reset())
            run_dir: Directory to save videos to
            num_episodes: Number of episodes to run per evaluation
            record_video: Whether to record videos
            video_every_calls: Record video every N evaluation calls (1 = every call)
        """
        self.env = env
        self.algo = algo
        self.run_dir = run_dir
        self.num_episodes = num_episodes
        self.record_video = record_video
        self.video_every_calls = video_every_calls
        self._eval_call_count = 0
        
        # Create videos directory
        if record_video:
            self.videos_dir = os.path.join(run_dir, 'videos')
            os.makedirs(self.videos_dir, exist_ok=True)
    
    def run(self, step: int) -> Dict[str, float]:
        """Evaluate algorithm for num_episodes episodes.
        
        Args:
            step: Global training step (for logging/video naming)
            
        Returns:
            Dictionary with 'return_mean' and 'return_std'
        """
        self._eval_call_count += 1
        
        returns = []
        lengths = []
        
        for episode_idx in range(self.num_episodes):
            # Decide whether to record this episode
            record_this = (
                self.record_video 
                and ((self._eval_call_count - 1) % self.video_every_calls == 0)
                and hasattr(self.env, 'render_mode')  # Only if env supports rendering
            )
            
            # Run one episode
            episode_return, episode_length, frames = self._run_episode(record=record_this)
            
            returns.append(episode_return)
            lengths.append(episode_length)
            
            # Save video if recorded
            if record_this and frames:
                video_path = os.path.join(
                    self.videos_dir,
                    f'step_{step}_episode_{episode_idx}.mp4'
                )
                self._save_video(frames, video_path)
        
        return {
            'return_mean': float(np.mean(returns)),
            'return_std': float(np.std(returns)),
            'length_mean': float(np.mean(lengths)),
            'length_std': float(np.std(lengths)),
        }
    
    def _run_episode(self, record: bool = False) -> tuple:
        """Run a single deterministic episode.
        
        Args:
            record: Whether to record frames
            
        Returns:
            Tuple of (episode_return, episode_length, frames_list)
        """
        frames = [] if record else None
        obs, _ = self.env.reset()
        
        # Reset any RNN state in the algorithm
        self.algo.reset()
        
        episode_return = 0.0
        episode_length = 0
        done = False
        
        while not done:
            # Record frame if needed
            if record:
                frame = self.env.render()
                if frame is not None:
                    frames.append(frame)
            
            # Get deterministic action
            action = self.algo.act(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            episode_return += float(reward)
            episode_length += 1
        
        return episode_return, episode_length, frames
    
    def _save_video(self, frames: list, path: str, fps: int = 30) -> None:
        """Save frames as a video file.
        
        Args:
            frames: List of RGB frames (H, W, 3)
            path: Path to save video to
            fps: Frames per second
        """
        if len(frames) == 0:
            return
        
        frame = np.asarray(frames[0])
        height, width = frame.shape[:2]

        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is not None:
            input_frames = [self._frame_to_rgb_uint8(frame) for frame in frames]
            raw_video = b''.join(frame.tobytes() for frame in input_frames)
            command = [
                ffmpeg,
                '-y',
                '-loglevel', 'error',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'rgb24',
                '-s', f'{width}x{height}',
                '-r', str(fps),
                '-i', 'pipe:0',
                '-an',
                '-vcodec', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                path,
            ]
            subprocess.run(command, input=raw_video, check=True)
            return

        try:
            import cv2
        except ImportError:
            raise RuntimeError('Neither ffmpeg nor opencv-python is available for video saving')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        try:
            for frame in frames:
                writer.write(cv2.cvtColor(self._frame_to_rgb_uint8(frame), cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

    def _frame_to_rgb_uint8(self, frame: np.ndarray) -> np.ndarray:
        """Convert a frame to uint8 RGB for encoding."""
        array = np.asarray(frame)
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                array = (array * 255.0).round()
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array
