from dataclasses import dataclass


@dataclass
class TrainConfig:

    batch_size: int = 32
    epochs: int = 30
    grad_accumulation_steps: int = 1
    gradient_clip_norm: float = 1.0
    label_smoothing: float = 0.1
    
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 4000
    
    monitor_metric: str = "loss" 
    monitor_mode: str = "min"
    
    checkpoint_dir: str = "weights"
    save_best_only: bool = True
    
    device: str = "cuda"
    num_workers: int = 0
    seed: int = 42