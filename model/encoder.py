from typing import Optional, Tuple

import torch
import torch.nn as nn

from model.layers import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1, 
        activation: str = "relu"
    ) -> None:
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        
        self.self_attention = MultiHeadAttention(
            d_model=d_model, 
            num_heads=num_heads,
            dropout=dropout
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model, 
            d_ff=d_ff, 
            dropout=dropout, 
            activation=activation
        )
        
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self_attention_output, self_attention_scores = self.self_attention(
            q=x, k=x, v=x, mask=src_mask
        )
        x = self.norm_1(x + self.dropout_1(self_attention_output))
        
        feed_forward_output = self.feed_forward(x=x)
        x = self.norm_2(x + self.dropout_2(feed_forward_output))
        
        return x, self_attention_scores 


class Encoder(nn.Module):
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu"
    ) -> None:
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_layers)
        ])
    
    def forward(
        self, 
        x: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self_attention_maps = []

        for layer in self.layers:
            x, self_attention_scores = layer(x, src_mask)
            self_attention_maps.append(self_attention_scores)

        return x, torch.stack(self_attention_maps, dim=0)