from typing import Optional, Tuple

import torch
import torch.nn as nn

from model.layers import (
    MultiHeadAttention,
    PositionwiseFeedForward
)


class DecoderLayer(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu"
    ) -> None:
        super(DecoderLayer, self).__init__()
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
        self.cross_attention = MultiHeadAttention(
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
        self.norm_3 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
    def forward(
        self,
        y: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self_attention_output, self_attention_scores = self.self_attention(
            q=y, k=y, v=y, mask=tgt_mask
        )
        y = self.norm_1(y + self.dropout_1(self_attention_output))
        
        cross_attention_output, cross_attention_scores = self.cross_attention(
            q=y, k=encoder_output, v=encoder_output, mask=src_mask
        )
        y = self.norm_2(y + self.dropout_2(cross_attention_output))
        
        feed_forward_output = self.feed_forward(x=y)
        y = self.norm_3(y + self.dropout_3(feed_forward_output))
        
        return y, self_attention_scores, cross_attention_scores
    
    
class Decoder(nn.Module):
    
    def __init__(
        self, 
        num_layers: int,
        num_heads: int,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu"
    ) -> None:
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        
        self.layers = nn.ModuleList([
            DecoderLayer(
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
        y: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self_attention_maps = []
        cross_attention_maps = []
        
        for layer in self.layers:
            y, self_attention_scores, cross_attention_scores = layer(
                y=y,
                encoder_output=encoder_output,
                src_mask=src_mask,
                tgt_mask=tgt_mask
            )
            self_attention_maps.append(self_attention_scores)
            cross_attention_maps.append(cross_attention_scores)
        
        return (
            y, 
            torch.stack(self_attention_maps, dim=0), 
            torch.stack(cross_attention_maps, dim=0)
        )