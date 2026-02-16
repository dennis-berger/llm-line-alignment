"""
Checkpoint utilities for resumable evaluation runs.

Supports saving and loading progress to handle quota limits and job interruptions.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalCheckpoint:
    """
    Checkpoint for evaluation progress.
    
    Attributes:
        method: Evaluation method (m1, m2, m3)
        dataset: Dataset name/path
        model: Model ID
        n_shots: Number of few-shot examples
        processed_ids: List of sample IDs that have been fully processed
        rows: Accumulated evaluation rows (list of lists)
        sums: Dictionary of accumulated metric sums for macro averaging
        checkpoint_path: Path where this checkpoint is stored
    """
    method: str
    dataset: str
    model: str
    n_shots: int = 0
    processed_ids: List[str] = field(default_factory=list)
    rows: List[List[Any]] = field(default_factory=list)
    sums: Dict[str, float] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None
    
    def save(self, path: Optional[Path] = None):
        """Save checkpoint to disk."""
        save_path = path or (Path(self.checkpoint_path) if self.checkpoint_path else None)
        if not save_path:
            raise ValueError("No checkpoint path specified")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = asdict(self)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        self.checkpoint_path = str(save_path)
        logger.info(f"Checkpoint saved: {save_path} ({len(self.processed_ids)} samples processed)")
    
    @classmethod
    def load(cls, path: Path) -> Optional["EvalCheckpoint"]:
        """Load checkpoint from disk, or return None if not found."""
        if not path.exists():
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            checkpoint = cls(**data)
            checkpoint.checkpoint_path = str(path)
            logger.info(f"Checkpoint loaded: {path} ({len(checkpoint.processed_ids)} samples already processed)")
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {path}: {e}")
            return None
    
    def mark_processed(self, sample_id: str, row: List[Any], metric_updates: Dict[str, float]):
        """Mark a sample as processed and update accumulators."""
        self.processed_ids.append(sample_id)
        self.rows.append(row)
        for key, value in metric_updates.items():
            self.sums[key] = self.sums.get(key, 0.0) + value
    
    def is_processed(self, sample_id: str) -> bool:
        """Check if a sample has already been processed."""
        return sample_id in self.processed_ids
    
    def delete(self):
        """Delete the checkpoint file (call when evaluation completes successfully)."""
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            Path(self.checkpoint_path).unlink()
            logger.info(f"Checkpoint deleted: {self.checkpoint_path}")


def get_checkpoint_path(
    method: str,
    dataset: str,
    model: str,
    n_shots: int = 0,
    checkpoint_dir: str = "checkpoints",
) -> Path:
    """
    Generate a deterministic checkpoint path for a given evaluation run.
    
    Args:
        method: Evaluation method (m1, m2, m3)
        dataset: Dataset name (e.g., 'bullinger_handwritten')
        model: Model ID (e.g., 'gemini/gemini-3-pro-preview')
        n_shots: Number of few-shot examples
        checkpoint_dir: Directory to store checkpoints
    
    Returns:
        Path to checkpoint file
    """
    # Sanitize model name for filename
    model_safe = model.replace("/", "_").replace(":", "_")
    dataset_safe = Path(dataset).name  # Get just the folder name
    
    filename = f"checkpoint_{method}_{dataset_safe}_{model_safe}_{n_shots}shot.json"
    return Path(checkpoint_dir) / filename
