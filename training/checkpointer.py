from typing import Optional, Dict, Any
from pathlib import Path

import torch

from model.transformer import Transformer


class Checkpointer:

    def __init__(
        self,
        checkpoint_dir: str,
        monitor: str = "val_loss",
        mode: str = "min",  # "min" or "max"
        save_best_only: bool = True,
    ):
        assert mode in {"min", "max"}

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only

        self.best_score = float("inf") if mode == "min" else -float("inf")

    def _is_better(self, score: float) -> bool:
        if self.mode == "min":
            return score < self.best_score
        return score > self.best_score

    def save(
        self,
        epoch: int,
        metrics: Dict[str, float],
        model: Transformer,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ):
        score = metrics.get(self.monitor)
        assert score is not None, f"Metric '{self.monitor}' not found in metrics"

        is_best = self._is_better(score)

        if self.save_best_only and not is_best:
            return

        if is_best:
            self.best_score = score

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "best_score": self.best_score,
        }

        last_path = self.checkpoint_dir / "last.pt"
        torch.save(state, last_path)

        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(state, best_path)