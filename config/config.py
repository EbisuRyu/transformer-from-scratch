import yaml
from dataclasses import dataclass


@dataclass
class ModelConfig:
    
    src_vocab_size: int = 16000
    tgt_vocab_size: int = 12000

    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048

    max_seq_len: int = 256
    dropout: float = 0.1
    activation: str = "gelu"

    pad_id: int = 0


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
    seed: int = 42


def load_config(
    train_path="config/train.yaml", 
    model_path="config/model.yaml"
):
    with open(train_path, 'r') as f:
        train_dict = yaml.safe_load(f)
    with open(model_path, 'r') as f:
        model_dict = yaml.safe_load(f)

    train_config = TrainConfig(**train_dict)
    model_config = ModelConfig(**model_dict)
    return train_config, model_config
