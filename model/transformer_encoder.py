# transformer_encoder.py

import torch
import torch.nn as nn

class ProteinTransformerEncoder(nn.Module):
    def __init__(self, 
                 d_model: int = 256,
                 nhead: int = 4,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 关键：让输入支持 (B, L, d_model)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src: torch.Tensor, src_key_padding_mask=None):
        """
        src: Tensor of shape (B, L, d_model)
        src_key_padding_mask: optional BoolTensor of shape (B, L) indicating which tokens should be masked
        """
        return self.encoder(src, src_key_padding_mask=src_key_padding_mask)  # (B, L, d_model)
