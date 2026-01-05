import torch
import torch.nn as nn

from config.config import ModelConfig
from model.encoder import Encoder
from model.decoder import Decoder
from model.layers import EmbeddingLayer
from utils.masks import make_pad_mask, make_causal_mask


class Transformer(nn.Module):
    
    def __init__(self, config: ModelConfig) -> None:
        super(Transformer, self).__init__()
        self.config = config
        
        self.src_embedding = EmbeddingLayer(
            d_model=config.d_model,
            vocab_size=config.src_vocab_size,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            positional_encoding_mode="sinusoidal"
        )
        self.tgt_embedding = EmbeddingLayer(
            d_model=config.d_model,
            vocab_size=config.tgt_vocab_size,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            positional_encoding_mode="sinusoidal"
        )
        
        self.encoder = Encoder(
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.activation
        )
        self.decoder = Decoder(
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.activation
        )
        
        self.lm_head = nn.Linear(config.d_model, config.tgt_vocab_size)
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor
    ):
        device = src.device
        batch_size, tgt_len = tgt.size()
        
        src_mask = make_pad_mask(src, pad_id=self.config.pad_id)  # (B, 1, 1, S)
        tgt_pad_mask = make_pad_mask(tgt, pad_id=self.config.pad_id)  # (B, 1, 1, T)
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