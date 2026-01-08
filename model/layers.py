import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        mode: Literal["sinusoidal", "learnable"] = "sinusoidal",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mode = mode
        self.dropout = nn.Dropout(dropout)

        if mode == "sinusoidal":
            pos = torch.arange(max_seq_len).float().unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

            pe = torch.zeros(max_seq_len, d_model)
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        else: 
            self.pe = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)

        if self.mode == "sinusoidal":
            x = x + self.pe[:, :seq_len].to(x.device)
        else:
            pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
            x = x + self.pe(pos)

        return self.dropout(x)


class EmbeddingLayer(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        max_seq_len: int,
        dropout: float = 0.1,
        positional_encoding_mode: str = "sinusoidal"
    ) -> None:
        super(EmbeddingLayer, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_seq_len=max_seq_len,
            mode=positional_encoding_mode,
            dropout=dropout
        )
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x) * (self.d_model ** 0.5)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        return x


class PositionwiseFeedForward(nn.Module):
    
    def __init__(
        self, 
        d_model: int, 
        d_ff: int, 
        dropout: float = 0.1,
        activation: str = "relu"
    ) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_head: int, dropout: float = 0.1):
        super().__init__()
        self.scale = math.sqrt(d_head)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        return out, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.dropout = dropout
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention(self.d_head, self.dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split(self, x):
        b, s, _ = x.size()
        x = x.view(b, s, self.num_heads, self.d_head)
        return x.transpose(1, 2).contiguous()

    def _combine(self, x):
        x = x.transpose(1, 2).contiguous()
        b, s, h, d_h = x.size()
        return x.view(b, s, h * d_h)

    def forward(self, q, k, v, mask=None):
        q = self._split(self.q_proj(q))
        k = self._split(self.k_proj(k))
        v = self._split(self.v_proj(v))

        out, attn = self.attn(q, k, v, mask)
        out = self._combine(out)
        out = self.out_proj(out)

        return out, attn