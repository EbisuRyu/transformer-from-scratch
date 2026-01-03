import torch
import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder
from model.layers import EmbeddingLayer
from model.utils import make_pad_mask, make_causal_mask


class Transformer(nn.Module):
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        activation: str = "relu",
        pad_id: int = 0
    ) -> None:
        super(Transformer, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.activation = activation
        self.pad_id = pad_id
        
        self.src_embedding = EmbeddingLayer(
            d_model=d_model,
            vocab_size=src_vocab_size,
            max_seq_len=max_seq_len,
            dropout=dropout,
            positional_encoding_mode="sinusoidal"
        )
        self.tgt_embedding = EmbeddingLayer(
            d_model=d_model,
            vocab_size=tgt_vocab_size,
            max_seq_len=max_seq_len,
            dropout=dropout,
            positional_encoding_mode="sinusoidal"
        )
        
        self.encoder = Encoder(
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )
        self.decoder = Decoder(
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )
        
        self.lm_head = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor
    ):
        device = src.device
        batch_size, tgt_len = tgt.size()
        
        src_mask = make_pad_mask(src, pad_id=self.pad_id)  # (B, 1, 1, S)
        tgt_pad_mask = make_pad_mask(tgt, pad_id=self.pad_id)  # (B, 1, 1, T)
        causal_mask = make_causal_mask(tgt_len, device)  # (1, 1, T, T)
        tgt_mask = tgt_pad_mask & causal_mask  # (B, 1, T, T)
        
        src_embeddings = self.src_embedding(src)
        tgt_embeddings = self.tgt_embedding(tgt)
        
        encoder_output, encoder_self_attention_maps = self.encoder(
            x=src_embeddings,
            src_mask=src_mask
        )
        
        decoder_output, decoder_self_attention_maps, decoder_cross_attention_maps = self.decoder(
            y=tgt_embeddings,
            encoder_output=encoder_output,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
        
        logits = self.lm_head(decoder_output)
        
        return {
            "logits": logits,
            "encoder_self_attention_maps": encoder_self_attention_maps,
            "decoder_self_attention_maps": decoder_self_attention_maps,
            "decoder_cross_attention_maps": decoder_cross_attention_maps
        }