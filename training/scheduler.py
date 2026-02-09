from typing import List

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer


class WarmupScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        last_epoch: int = -1,
    ) -> None:
        self.num_steps = 0          
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def step(self) -> None:
        self.num_steps += 1
        return super().step()

    def get_lr(self) -> List[float]:
        step = max(self.num_steps, 1)
        scale = (
            self.d_model ** -0.5
            * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        )
        return [base_lr * scale for base_lr in self.base_lrs]