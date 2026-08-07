"""Logging wrapper around wandb for the RL suite."""

import wandb
from typing import Any, Dict, Optional
import os


class Logger:
    """Wrapper around Weights & Biases for experiment tracking."""
    
    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        run_dir: Optional[str] = None,
    ):
        """Initialize logger.
        
        Args:
            project: WandB project name
            entity: WandB entity/team name (optional)
            run_name: Human-readable run name
            config: Config dictionary to log
            run_dir: Directory to save run artifacts
        """
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.config = config or {}
        self.run_dir = run_dir

        init_mode = os.environ.get("WANDB_MODE")
        if init_mode is None and not os.environ.get("WANDB_API_KEY"):
            init_mode = "offline"
        
        # Initialize wandb
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=self.config,
            dir=run_dir,
            mode=init_mode,
        )
    
    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metric_name -> value
            step: Step/iteration number (optional)
        """
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)
    
    def log_scalar(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single scalar metric.
        
        Args:
            key: Metric name (e.g. "train/loss")
            value: Metric value
            step: Step number (optional)
        """
        self.log({key: value}, step=step)
    
    def log_dict(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log a dictionary of metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Step number (optional)
        """
        self.log(metrics, step=step)
    
    def finish(self) -> None:
        """Finalize the run."""
        if self.run is not None:
            wandb.finish()
    
    def save_file(self, filepath: str) -> None:
        """Save a file to the run directory with wandb.
        
        Args:
            filepath: Path to file to save
        """
        if self.run is not None:
            wandb.save(filepath)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get the run summary.
        
        Returns:
            Dictionary of final values
        """
        return dict(self.run.summary)
