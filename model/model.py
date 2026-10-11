from json import decoder
import torch
import torch.nn as nn

from .tokenizer import ProteinTokenizer
from .embedding import CombinedProteinEmbedding
from .decoder import ProteinDecoderOnlyModel, ProteinMaskPredictModel
import os
from typing import Union
import numpy as np

class ProteinLanguageModel(nn.Module):
    def __init__(self, 
                 tokenizer: ProteinTokenizer=None,
                 embedding: CombinedProteinEmbedding=None,
                 decoder: Union[ProteinDecoderOnlyModel, ProteinMaskPredictModel]=None,
                 d_model: int=256, 
                 hidden_dim: int = 64, 
                 dim_feedforward: int = 512, 
                 nhead=8, 
                 mode: str = "sum",
                 num_layers: int = 2, 
                 dropout: float = 0.1, 
                 decoder_use_cls: bool = True,
                 decoder_cls_cat_feat: bool = True,
                 decoder_cls_global_predict: bool = False
                 ):    
        super().__init__()
        self.tokenizer = tokenizer if tokenizer else ProteinTokenizer([])
        if embedding is None:
            self.embedding = CombinedProteinEmbedding(
                num_proteins=self.tokenizer.vocab_size,
                d_model=d_model,
                hidden_dim=hidden_dim,
                mode=mode
            )
        else:
            self.embedding = embedding
        if decoder is None:
            self.decoder = ProteinMaskPredictModel(
                vocab_size=self.tokenizer.vocab_size,
                embedding=self.embedding,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                use_cls=decoder_use_cls,
                cls_cat_feat=decoder_cls_cat_feat,
                cls_global_predict=decoder_cls_global_predict
            )
        else:
            self.decoder = decoder
            
        self.decoder_use_cls = decoder_use_cls
        self.decoder_cls_cat_feat = decoder_cls_cat_feat
        self.decoder_cls_global_predict = decoder_cls_global_predict
        
    def forward(self, input_ids, expr_values=None, pos_ids=None, mask=None,  **kwargs):
        """
        input_ids: (B, L) long
        expr_values: (B, L) float
        pos_ids:     (B, L) long
        """
        
        return self.decoder(input_ids=input_ids, expr_values=expr_values, pos_ids=pos_ids, mask=mask, **kwargs)
    
    def save_model(self, path: str):
        """
        Save model state, config, and tokenizer to checkpoint.
        """
        assert os.path.exists(os.path.dirname(path)), f"Directory {os.path.dirname(path)} does not exist."
        config = {
            "d_model": self.decoder.d_model,
            "hidden_dim": self.embedding.expr_embed.mlp[0].out_features,
            "dim_feedforward": self.decoder.encoder.layers[0].linear1.out_features,
            "nhead": self.decoder.encoder.layers[0].self_attn.num_heads,
            "num_layers": len(self.decoder.encoder.layers),
            "dropout": self.decoder.encoder.layers[0].dropout.p,
            "mode": self.embedding.mode,
            "use_cls": self.decoder.use_cls,
            "cls_cat_feat": self.decoder.cls_cat_feat,
            "cls_global_predict": self.decoder.cls_global_predict
        }

        # 提取tokenizer配置
        tokenizer_config = {
            "protein_names": [k for k in self.tokenizer.protein_to_id if k not in self.tokenizer.special_tokens],
            "special_tokens": self.tokenizer.special_tokens
        }

        save_dict = {
            "model_state_dict": self.state_dict(),
            "config": config,
            "tokenizer_config": tokenizer_config
        }

        torch.save(save_dict, path)
        print(f"Model saved to {path}")
    
    @staticmethod
    def load_model_from_checkpoint(path: str, device: str = "cpu") -> "ProteinLanguageModel":
        """
        Load model and tokenizer from a checkpoint with config.
        Returns: ProteinLanguageModel instance
        """
        assert os.path.exists(path), f"Checkpoint {path} does not exist."
        checkpoint = torch.load(path, map_location=device)
        
        config = checkpoint["config"]
        tokenizer = ProteinTokenizer(**checkpoint["tokenizer_config"])

        model = ProteinLanguageModel(
            tokenizer=tokenizer,
            d_model=config["d_model"],
            hidden_dim=config["hidden_dim"],
            dim_feedforward=config["dim_feedforward"],
            nhead=config["nhead"],
            mode=config["mode"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            decoder_use_cls=config["use_cls"],
            decoder_cls_cat_feat=config["cls_cat_feat"],
            decoder_cls_global_predict=config["cls_global_predict"]
        )
        
        model.load_state_dict(checkpoint["model_state_dict"])

        return model
        
    @torch.no_grad() 
    def get_cell_embedding(self, input_ids, expr_values, pos_ids=None, method: str="cls"):
        """
        根据 decoder 输出和表达量，计算每个细胞的 embedding。
        
        Args:
            protein_ids: (B, L) long tensor
            expr_values: (B, L) float tensor, CLR 变换后的表达量
            pos_ids:     (B, L) long tensor
            method:      str, one of ["exp_weighted", "mean", "minmax"]
            
        Returns:
            cell_embedding: (B, d_model) tensor
        """
        self.eval()

        with torch.no_grad():
            if method == "cls":
                cell_embedding = self.decoder(input_ids=input_ids, expr_values=expr_values, pos_ids=pos_ids, return_cls=True)
            elif method == "mean":
                x = self.decoder(input_ids, expr_values, pos_ids)  # (B, L, d_model)
                cell_embedding = x.mean(dim=1)  # (B, d_model)

        return cell_embedding

    def add_new_proteins(self, new_proteins: list):
        """
        增加新的蛋白质到tokenizer，并调整embedding层大小。
        """
        old_vocab_size = self.tokenizer.vocab_size
        added = self.tokenizer.add_tokens(new_proteins)
        if added > 0:
            new_vocab_size = self.tokenizer.vocab_size
            self.embedding.resize_protein_embedding(new_vocab_size)
            print(f"Added {added} new proteins. Resized embeddings to vocab size {new_vocab_size}.")
            if self.decoder_cls_global_predict:
                self.decoder.resize_cls_global_head(new_vocab_size)
                print(f"Resized decoder cls global head to vocab size {new_vocab_size}.")
        else:
            print("No new proteins added. Tokenizer and embeddings remain unchanged.")
            
        return added, old_vocab_size
 
    def get_protein_embedding(self, input_ids, expr_values=None, pe_index=None, mask=None):
        return self.decoder.get_protein_embedding(input_ids, expr_values, pe_index, mask)
    
    def get_cls_cat_feat(self, input_ids, expr_values=None, pe_index=None, mask=None):
        return self.decoder.get_cls_cat_feat(input_ids, expr_values, pe_index, mask)
            
if __name__ == "__main__":
    model_path = "/home/hubowen/work/protein_llm/checkpoints/ClsGlo+lr5e-4+weidec1e-3/2025-10-19_02-14-25/model_140.pt"
    model = ProteinLanguageModel.load_model_from_checkpoint(model_path)
    print(model.tokenizer.protein_to_id)
    print(model.decoder.cls_head.out_features)
    model.add_new_proteins(["NEW1", "NEW2", "CD45"])
    print(model.tokenizer.protein_to_id)
    print(model.decoder.cls_head.out_features)
                                            