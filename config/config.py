import yaml
from dataclasses import dataclass


@dataclass
class ModelConfig:
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    dropout_rate: float
    max_seq_len: int
    vocab_size: int
    activation: str
    layer_norm_eps: float


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-9
    warmup_steps: int = 4000
    optimizer: str = "adam"           
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    label_smoothing: float = 0.1
    grad_accumulation_steps: int = 1     
    model_save_epoch_cnt: int = 1          
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
