# decoder.py

from os import name
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)

class ProteinDecoderOnlyModel(nn.Module):
    def __init__(self, 
                 vocab_size,
                 embedding: nn.Module,
                 d_model=256, 
                 nhead=8, 
                 num_layers=4, 
                 dim_feedforward=512, 
                 dropout=0.1):
        super().__init__()

        self.embedding = embedding
        self.d_model = d_model

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # === 分类分支 ===
        self.cls_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.vocab_embed = nn.Parameter(torch.randn(vocab_size, d_model))
        nn.init.xavier_uniform_(self.vocab_embed)

        # === 回归分支 ===
        self.reg_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.expr_head = nn.Linear(d_model, 1)

    def forward(self, protein_ids, expr_values, pos_ids, mask=None):
        B, L = protein_ids.size()
        device = protein_ids.device

        tgt_mask = self._generate_causal_mask(L, device)

        # 基础特征
        x = self.embedding(protein_ids, expr_values, pos_ids)
        dummy_memory = torch.zeros(B, 1, self.d_model, device=device)
        x = self.decoder(tgt=x, memory=dummy_memory, tgt_mask=tgt_mask)

        # === 分类路径 ===
        cls_feat = self.cls_proj(x)  # (B, L, d_model)
        cls_feat_norm = F.normalize(cls_feat, dim=-1)
        vocab_embed_norm = F.normalize(self.vocab_embed, dim=-1)
        name_logits = torch.matmul(cls_feat_norm, vocab_embed_norm.t()) * 10.0  # 温度可调

        # === 回归路径 ===
        reg_feat = self.reg_proj(x)  # (B, L, d_model)
        expr_pred = self.expr_head(reg_feat).squeeze(-1)

        return name_logits, expr_pred

    def _generate_causal_mask(self, size, device):
        mask = torch.triu(torch.full((size, size), float('-inf'), device=device), diagonal=1)
        return mask
        
class ProteinMaskPredictModel(nn.Module):
    def __init__(self, 
                 vocab_size,
                 embedding: nn.Module,
                 d_model=256, 
                 nhead=8, 
                 num_layers=4, 
                 dim_feedforward=512, 
                 dropout=0.1,
                 use_cls: bool = True,
                 cls_cat_feat: bool = True,
                 cls_global_predict: bool = False):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding = embedding
        self.d_model = d_model
        self.use_cls = use_cls
        self.cls_cat_feat = cls_cat_feat
        self.cls_global_predict = cls_global_predict
        
        in_dim = d_model

        if use_cls:
            # === CLS token ===
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # (1, 1, d_model)
            # === CLS 处理分支 ===
            self.cls_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model)
            )
            if cls_global_predict:
                self.cls_global_head = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, self.vocab_size)
                )
            if cls_cat_feat:
                in_dim = d_model * 2  # 拼接 CLS 向量

        # === Transformer Encoder ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # === 回归分支 ===
        self.reg_proj = nn.Sequential(
            # 1. 第一步：升维。将 256 映射到 1024。
            nn.Linear(in_dim, d_model * 4),
            nn.LayerNorm(d_model * 4), # 必须在这里稳住分布
            nn.GELU(),
            
            # 2. 第二步：高维特征提取。
            nn.Linear(d_model * 4, d_model * 4),
            nn.LayerNorm(d_model * 4),
            nn.GELU(),
            
            # 3. 第三步：残差块思路（手动展开版）。
            # 这一步能有效缓解深层网络的“平均化预测”倾向
            nn.Linear(d_model * 4, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            
            # 4. 最后降维到输出维度
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        # self.reg_proj = nn.Sequential(
        #     nn.Linear(in_dim, d_model),
        #     nn.PReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_model, d_model),
        # )
        self.expr_head = nn.Linear(d_model, 1)

    def forward(self, input_ids, expr_values=None, pos_ids=None, mask=None, return_cls: bool=False, **kwargs):
        """
        input_ids: (B, L)
        expr_values: (B, L)
        pos_ids:     (B, L)
        mask:        (B, L) bool，True 表示该位置被掩码
        """
        if return_cls and not self.use_cls:
            raise ValueError("Model is not using CLS token, cannot return CLS embedding.")
        B, L = input_ids.size()
        device = input_ids.device

        # === Step 1: 获取 embedding ===
        x = self.embedding(input_ids, expr_values, pos_ids, mask=mask)  # (B, L, d_model)

        if self.use_cls:
            # === Step 2: 拼接 CLS token ===
            cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
            x = torch.cat([cls_token, x], dim=1)  # (B, L+1, d_model)

        # === Step 3: Transformer 编码 ===
        x = self.encoder(x)  # (B, L+1, d_model)

        global_pred = None
        expr_pred = None

        if self.use_cls:
            # CLS 向量
            cls_feat = self.cls_proj(x[:, 0])  # (B, d_model)
            if return_cls:
                return cls_feat

            # 每个蛋白的 hidden state
            seq_feat = x[:, 1:]  # (B, L, d_model)

            # === 回归路径 ===
            if self.cls_cat_feat:
                cls_expand = cls_feat.unsqueeze(1).expand(-1, L, -1)  # (B, L, d_model)
                reg_input = torch.cat([seq_feat, cls_expand], dim=-1)  # (B, L, 2*d_model)
            else:
                reg_input = seq_feat  # (B, L, d_model)
                
            if self.cls_global_predict:
                global_pred = self.cls_global_head(cls_feat)  # (B, vocab_size)
            
        else:
            seq_feat = x  # (B, L, d_model)
            reg_input = seq_feat  # (B, L, d_model)

        reg_feat = self.reg_proj(reg_input)  # (B, L, d_model)
        expr_pred = self.expr_head(reg_feat).squeeze(-1)  # (B, L)

        # 如果有 mask，就只返回 mask 部分
        if mask is not None:
            expr_pred = expr_pred[mask]

        return expr_pred, global_pred

    def resize_cls_global_head(self, new_vocab_size):
        """
        扩展 CLS global head 的输出层到新的 vocab_size。
        """
        if not hasattr(self, "cls_global_head"):
            raise ValueError("Model does not have cls_global_head defined.")

        # 获取旧的最后一层 Linear
        old_linear = self.cls_global_head[-1]
        old_out_features = old_linear.out_features
        in_features = old_linear.in_features

        if new_vocab_size == old_out_features:
            print(f"[resize_cls_global_head] vocab_size unchanged ({new_vocab_size})")
            return

        # print(f"[resize_cls_global_head] expanding vocab from {old_out_features} to {new_vocab_size}")

        # === 新建一层 Linear ===
        new_linear = nn.Linear(in_features, new_vocab_size)
        nn.init.xavier_uniform_(new_linear.weight)
        nn.init.zeros_(new_linear.bias)

        # === 复制旧参数到新层 ===
        with torch.no_grad():
            num_to_copy = min(old_out_features, new_vocab_size)
            new_linear.weight[:num_to_copy, :] = old_linear.weight[:num_to_copy, :]
            new_linear.bias[:num_to_copy] = old_linear.bias[:num_to_copy]

        # === 替换掉旧层 ===
        self.cls_global_head[-1] = new_linear

        # 同步 vocab_size 记录
        self.vocab_size = new_vocab_size

    def get_protein_embedding(self, input_ids, expr_values=None, pos_ids=None, mask=None):
        B, L = input_ids.size()

        x = self.embedding(input_ids, expr_values, pos_ids, mask=mask)  # (B, L, d_model)

        if self.use_cls:
            cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
            x = torch.cat([cls_token, x], dim=1)  # (B, L+1, d_model)

        x = self.encoder(x)  # (B, L+1, d_model)
        seq_feat = x[:, 1:]  # (B, L, d_model)
        protein_embedding = self.reg_proj(seq_feat)

        return protein_embedding
        
    def get_cls_cat_feat(self, input_ids, expr_values=None, pos_ids=None, mask=None, **kwargs):
        B, L = input_ids.size()

        # === Step 1: 获取 embedding ===
        x = self.embedding(input_ids, expr_values, pos_ids, mask=mask)  # (B, L, d_model)

        if self.use_cls:
            # === Step 2: 拼接 CLS token ===
            cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
            x = torch.cat([cls_token, x], dim=1)  # (B, L+1, d_model)

        # === Step 3: Transformer 编码 ===
        x = self.encoder(x)  # (B, L+1, d_model)

        cls_feat = self.cls_proj(x[:, 0])  # (B, d_model)

        # 每个蛋白的 hidden state
        seq_feat = x[:, 1:]  # (B, L, d_model)

        cls_expand = cls_feat.unsqueeze(1).expand(-1, L, -1)  # (B, L, d_model)
        reg_input = torch.cat([seq_feat, cls_expand], dim=-1)  # (B, L, 2*d_model)
        
        return reg_input