from typing import Dict, Type, Any, Tuple, Optional
from dataclasses import dataclass
import yaml


@dataclass
class ModelConfig:
    
    src_vocab_size: int
    tgt_vocab_size: int

    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    positional_encoding_mode: str

    max_seq_len: int
    dropout: float
    activation: str

    pad_id: int
    

@dataclass
class TrainConfig:

    batch_size: int
    epochs: int
    grad_accumulation_steps: int
    gradient_clip_norm: float
    label_smoothing: float
    
    optimizer: str
    learning_rate: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    
    warmup_steps: Optional[int]
    
    monitor_metric: str
    monitor_mode: str
    
    checkpoint_dir: str
    save_best_only: bool
    
    device: str
    num_workers: int
    seed: int
    

@dataclass
class EvaluateConfig:
    
    src_file: str
    ref_file: str
    
    checkpoint: str
    device: str
    
    bert_model: str
    max_seq_len: str
    strategy: str
    beam_size: int


def load_config_from_yaml(config_type: str, file_path: str) -> Any:
    config_registry: Dict[str, Type] = {
        "model": ModelConfig,
        "train": TrainConfig,
        'evaluate': EvaluateConfig
    }

    if config_type not in config_registry:
        raise ValueError(
            f"Unknown config type '{config_type}'. "
            f"Available types: {list(config_registry.keys())}"
        )

    config_cls = config_registry[config_type]

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    try:
        return config_cls(**data)
    except TypeError as e:
        raise ValueError(
            f"Invalid YAML for config type '{config_type}' at {file_path}"
        ) from e
