from dataclasses import dataclass


@dataclass
class ModelConfig:
    
    src_vocab_size: int = 16000
    tgt_vocab_size: int = 12000

    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048
    positional_encoding_mode: str = "sinusoidal"

    max_seq_len: int = 256
    dropout: float = 0.1
    activation: str = "relu"

    pad_id: int = 0