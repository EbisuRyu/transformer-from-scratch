import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        mode: Literal["sinusoidal", "learnable"] = "sinusoidal"
    ) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.mode = mode
        self._generate_positional_encoding()
    
    def _generate_positional_encoding(self) -> None:
        if self.mode == "sinusoidal":
            position = torch.arange(0, self.max_seq_len, dtype=torch.float32).unsqueeze(1)
            _2i = torch.arange(0, self.d_model, step=2, dtype=torch.float32)
            self.positional_encoding = torch.zeros(self.max_seq_len, self.d_model, requires_grad=False)
            self.positional_encoding[:, 0::2] = torch.sin(position / torch.pow(10000, _2i / self.d_model))
            self.positional_encoding[:, 1::2] = torch.sin(position / torch.pow(10000, _2i / self.d_model))
            self.positional_encoding = self.positional_encoding
        
        elif self.mode == "learnable":
            self.positional_encoding = nn.Embedding(
                num_embeddings=self.max_seq_len,
                embedding_dim=self.d_model
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        x = x + self.positional_encoding[positions]
        return x


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
            mode=positional_encoding_mode
        )
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LongTensor of shape (batch_size, seq_len)
            
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
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
    
    def __init__(
        self, 
        d_head: int, 
        dropout: float = 0.1,
    ) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.d_head = d_head
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor]:
        """
        Args:
            q: Query tensor of shape (batch_size, num_heads, seq_len_q, d_head).
            k: Key tensor of shape (batch_size, num_heads, seq_len_k, d_head).
            v: Value tensor of shape (batch_size, num_heads, seq_len_k, d_head).
            mask: Optional attention mask.
                Shape can be:
                - (batch_size, 1, 1, seq_len_k) for padding mask, or
                - (batch_size, 1, seq_len_q, seq_len_k) for causal/self-attention.
                Mask values should be 0 for positions to mask and 1 otherwise.
        
        Returns:
            weighted_values: Attention output of shape (batch_size, num_heads, seq_len_q, d_head).
            attention_scores: Normalized attention weights after softmax, shape (batch_size, num_heads, seq_len_q, seq_len_k).
        """
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            attention_weights = attention_weights.masked_fill(
                mask == 0, float("-inf")
            )

        attention_scores = torch.softmax(attention_weights, dim=-1)
        attention_scores = self.dropout(attention_scores)
        weighted_values = torch.matmul(attention_scores, v)

        return weighted_values, attention_scores


class MultiHeadAttention(nn.Module):
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int,
        dropout: float = 0.1
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert self.d_model % num_heads == 0
        self.d_head = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(
            d_head=self.d_head
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def _split_into_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_head)
        return x.transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = x.reshape(batch_size, seq_len, -1)
        return x
    
    def forward(
        self, 
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Query tensor of shape (batch_size, num_heads, seq_len_q, d_head).
            k: Key tensor of shape (batch_size, num_heads, seq_len_k, d_head).
            v: Value tensor of shape (batch_size, num_heads, seq_len_k, d_head).
            mask: Optional attention mask.
                Shape can be:
                - (batch_size, 1, 1, seq_len_k) for padding mask, or
                - (batch_size, 1, seq_len_q, seq_len_k) for causal/self-attention.
                Mask values should be 0 for positions to mask and 1 otherwise.
        
        Returns:
            output: Attention output of shape (batch_size, num_heads, seq_len_q, d_head).
            scores: Normalized attention weights after softmax, shape (batch_size, num_heads, seq_len_q, seq_len_k).
        """
        q = self._split_into_heads(self.q_proj(q))
        k = self._split_into_heads(self.k_proj(k))
        v = self._split_into_heads(self.v_proj(v))
            
        output, scores = self.attention(q, k, v, mask)
        output = self._combine_heads(output)
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output, scores