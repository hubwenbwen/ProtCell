# embedding.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)

class ProteinEmbedding(nn.Module):
    def __init__(self, num_proteins: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(num_proteins, d_model)
        self.embedding_norm = nn.LayerNorm(d_model)
        
    def resize_embedding(self, new_vocab_size):
        old_weights = self.embedding.weight.data
        old_vocab_size, d_model = old_weights.shape
        if new_vocab_size > old_vocab_size:
            new_weights = torch.randn(new_vocab_size, d_model, device=old_weights.device)
            new_weights[:old_vocab_size] = old_weights
            self.embedding = nn.Embedding(new_vocab_size, d_model)
            self.embedding.weight.data = new_weights

    def forward(self, protein_ids: torch.Tensor):  # shape: (B, L)
        if torch.any(torch.isnan(protein_ids)) or torch.any(torch.isinf(protein_ids)):
            logger.warning(f"Protein IDs contain nan or inf: {protein_ids}")
        return self.embedding_norm(self.embedding(protein_ids))         # shape: (B, L, d_model)
        # return self.embedding(protein_ids)         # shape: (B, L, d_model)


class ExpressionEmbedding(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            # nn.LayerNorm(d_model)
        )
        self.mask_embedding = nn.Parameter(torch.randn(d_model))
        self.d_model = d_model

    def forward(self, expr_values: torch.Tensor, mask=None):  # shape: (B, L)
        x = self.mlp(expr_values.unsqueeze(-1))  # (B, L, d_model)
        # if torch.any(torch.isnan(expr_values)) or torch.any(torch.isinf(expr_values)):
        #     logger.warning(f"Expression values contain nan or inf: {expr_values}")
            
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (B, L, 1)
            x = torch.where(mask.expand(-1, -1, self.d_model),
                            self.mask_embedding.view(1, 1, -1),
                            x)

        return x  # (B, L, d_model)


class PositionEmbedding(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, pos_ids: torch.Tensor):  # shape: (B, L)
        pos_ids = pos_ids.float().unsqueeze(-1)  # (B, L, 1)
        return self.mlp(pos_ids)                 # (B, L, d_model)


class CombinedProteinEmbedding(nn.Module):
    def __init__(self, num_proteins: int, d_model: int, hidden_dim: int = 64, mode: str = "sum"):
        """
        mode: "sum" 或 "concat"
        """
        super().__init__()
        self.protein_embed = ProteinEmbedding(num_proteins, d_model)
        self.expr_embed = ExpressionEmbedding(d_model, hidden_dim)
        self.pos_embed = PositionEmbedding(d_model, hidden_dim)
        self.d_model = d_model

        self.mode = mode
        if mode == "concat":
            self.project = nn.Linear(d_model * 3, d_model)
            
    def resize_protein_embedding(self, new_vocab_size):
        self.protein_embed.resize_embedding(new_vocab_size)

    def forward(self, protein_ids, expr_values, pos_ids=None, mask=None):
        """
        protein_ids: (B, L) int
        expr_values: (B, L) float
        pos_ids:     (B, L) int
        """
        p = self.protein_embed(protein_ids)   # (B, L, d_model)
        e = self.expr_embed(expr_values, mask=mask)      # (B, L, d_model)
        # pos = self.pos_embed(pos_ids)         # (B, L, d_model)

        if self.mode == "sum":
            return p + e  # (B, L, d_model)
        elif self.mode == "concat":
            combined = torch.cat([p, e], dim=-1)  # (B, L, d_model * 2)
            return self.project(combined)              # (B, L, d_model)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

