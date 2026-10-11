import torch
import torch.nn.functional as F
from finetuner.BaseFineTuner import BaseFineTuner, unwrap_model
from typing import List
from torch import nn
import os
from torch.utils.data import Dataset, DataLoader
from anndata import AnnData

import torch
import torch.nn as nn
import os

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
    def forward(self, x):
        return x + self.net(x)

class P2RMLP(nn.Module):
    def __init__(self, in_dim: int, d_model: int, out_dim: int = 40000):
        super(P2RMLP, self).__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.out_dim = out_dim

        # 基础投影
        self.input_proj = nn.Linear(in_dim, d_model * 2)
        
        # 引入残差层来深化对蛋白质特征的理解
        self.res_layers = nn.Sequential(
            ResidualBlock(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            ResidualBlock(d_model)
        )

        # 分类头：预测 40000 个基因每个是否为非零（即是否表达）
        self.binary_head = nn.Sequential(
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_layers(x)
        return self.binary_head(x)

    def save_head(self, save_dir: str, file_name: str):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        checkpoint = {
            "state_dict": self.state_dict(),
            "params": {
                "in_dim": self.in_dim,
                "d_model": self.d_model,
                "out_dim": self.out_dim
            }
        }
        torch.save(checkpoint, save_path)

    @staticmethod
    def load_head(load_path: str, device: str = "cpu"):
        assert os.path.exists(load_path), f"File {load_path} does not exist."
        checkpoint = torch.load(load_path, map_location=device)
        p2r_mlp = P2RMLP(**checkpoint["params"])
        p2r_mlp.load_state_dict(checkpoint["state_dict"])
        return p2r_mlp
        

class ProteinSequenceDataset(Dataset):
    """
    Dataset for protein sequences without sorting or EOS appending.
    """
    def __init__(self, adata: AnnData, rna_data, tokenizer):
        self.adata = adata
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_token = tokenizer.pad_token
        self.rna_binary = torch.from_numpy(rna_data.X.toarray() > 0)

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        protein_names = list(self.adata.var_names)
        expr_values = self.adata.X[idx].toarray().flatten() if hasattr(self.adata.X[idx], 'toarray') else self.adata.X[idx]
        input_ids = [self.tokenizer.encode(name) for name in protein_names]
        

        expr_values, input_ids = zip(*[
            (v, i)
            for v, i in zip(expr_values, input_ids)
            if v >= 0
        ])
        expr_values, input_ids = list(expr_values), list(input_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        expr_values = torch.tensor(expr_values, dtype=torch.float)

        return {
            "input_ids": input_ids,       # (L,)
            "expr_values": expr_values,   # (L,)
            "idx": idx,
            "rna_target": self.rna_binary[idx],
        }

def collate_fn(batch):
    pad_token_id =  0 
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids, expr_values, attn_mask, idx_list = [], [], [], []
    rna_target = []

    for x in batch:
        L = len(x["input_ids"])
        pad_len = max_len - L

        input_ids.append(F.pad(x["input_ids"], (0, pad_len), value=pad_token_id))
        expr_values.append(F.pad(x["expr_values"], (0, pad_len), value=-1.0))
        attn_mask.append(torch.cat([torch.ones(L), torch.zeros(pad_len)]))
        idx_list.append(x["idx"])
        rna_target.append(x["rna_target"])

    return {
        "input_ids": torch.stack(input_ids),
        "expr_values": torch.stack(expr_values),
        "attention_mask": torch.stack(attn_mask),
        "idx": idx_list,
        "rna_target": torch.stack(rna_target)
    }


class P2RImputationTrainer(BaseFineTuner):
    def __init__(self,
                 RNA_data,
                 **kwargs):
        super().__init__(**kwargs)
        self.RNA_data = RNA_data
        self.dist_head = P2RMLP(in_dim=self.model.decoder.d_model,
                                        d_model=self.model.decoder.d_model * 2, 
                                        out_dim=self.RNA_data.n_vars).to(self.device)
        
        num_pos = (self.RNA_data.X > 0).sum().item()
        num_neg = (self.RNA_data.X == 0).sum().item()
        self.global_pos_weight = num_neg / num_pos if (num_pos > 0 and num_neg > 0) else 1.0
        
    def deploy_lora(self,
                    lora_rank: int = 8,
                    lora_alpha: int = 16,
                    lora_dropout: float = 0.05,
                    target_modules: List[str] = ["linear1", "linear2", "cls_proj.0", "cls_proj.3"]):
        
        from peft import get_peft_model, LoraConfig, TaskType
        core = unwrap_model(self.model)
                
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.FEATURE_EXTRACTION,
            bias="none"
        )

        core.decoder = get_peft_model(core.decoder, lora_config)
        
        core.decoder.print_trainable_parameters()
        # for name, param in self.model.named_parameters():
        #     if "norm" in name.lower():
        #         param.requires_grad = True
 
    def prepare_dataloader(self, **kwargs):
        if self.adata is None:
            raise ValueError("adata is not set for dataloader preparation")
        dataset = ProteinSequenceDataset(self.adata, self.RNA_data, self.tokenizer)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

    def on_fit_start(self, **kwargs):
        self.model.train()
        self.prepare_dataloader()
        
        if "lora" in self.trainable_modules:        
            lora_rank = kwargs.get("lora_rank", 8)
            lora_alpha = kwargs.get("lora_alpha", 16)
            lora_dropout = kwargs.get("lora_dropout", 0.05)
            target_modules = kwargs.get("target_modules", ["linear1", "linear2", "cls_proj.0", "cls_proj.3"])
            self.deploy_lora(lora_rank, lora_alpha, lora_dropout, target_modules)
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler(T_max=10)
     
    def _get_optimizer(self):
        """
        根据 trainable_modules 中的配置，为不同层分配不同的学习率
        """
        core = unwrap_model(self.model)
        core.requires_grad_(False)  # 先冻结所有参数，后面根据 trainable_modules 的配置逐个解冻
        parameter_groups = []
        # 记录已经分配过的参数，防止重复（例如某个层既属于 linear 又属于 mlp）
        assigned_params = set()

        # 1. 按照 trainable_modules 的定义进行分组
        # 按照 key (模块名关键词) 和 value (对应的学习率) 迭代
        for module_name_key, custom_lr in self.trainable_modules.items():
            group_params = []
            found = False
            for name, param in core.named_parameters():
                if module_name_key in name:
                    param.requires_grad = True
                    if id(param) not in assigned_params:
                        group_params.append(param)
                        assigned_params.add(id(param))
                        found = True

            if not found:
                print(f"[Warning] No parameters found for module key '{module_name_key}'.")
            if group_params:
                parameter_groups.append({
                    "params": group_params,
                    "lr": custom_lr,
                    "name": module_name_key  # 方便调试查看
                })

        if self.dist_head is not None:
            dist_head_params = list(self.dist_head.parameters())
            for p in dist_head_params:
                p.requires_grad = True
                
            parameter_groups.append({
                "params": dist_head_params, 
                "lr": self.lr 
            })

        if not parameter_groups:
            raise ValueError("No trainable parameters found. Check requires_grad or trainable_modules.")

        opt_name = self.optimizer_cls.lower() if isinstance(self.optimizer_cls, str) else ""
        
        if opt_name == 'adamw':
            return torch.optim.AdamW(parameter_groups, weight_decay=self.weight_decay)
        elif opt_name == 'adam':
            return torch.optim.Adam(parameter_groups, weight_decay=self.weight_decay)

        return self.optimizer_cls(parameter_groups)
 
    def save_model(self, save_dir: str = None, file_name: str = None, epoch: int = 0, step: int = 0, **kwargs):
        if self.expr_name:
            save_dir = os.path.join(save_dir, self.expr_name)
        os.makedirs(save_dir, exist_ok=True)
        import copy
        core = unwrap_model(self.model)
        core_copy = copy.deepcopy(core)
        
        # 在副本上合并LoRA
        if "lora" in self.trainable_modules:
            core_copy.decoder = core_copy.decoder.merge_and_unload()
        
        # 使用副本的save_model方法保存
        save_path = os.path.join(save_dir, f"base_{epoch}.pt")
        core_copy.save_model(save_path)
        
        # 删除副本释放内存
        del core_copy
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if self.dist_head is not None:
            self.dist_head.save_head(save_dir, f"head_{epoch}.pt")
            print(f"Saved distribution head to {os.path.join(save_dir, file_name)}")
    
    def compute_loss(self, batch, **kwargs):
        input_ids = batch["input_ids"].to(self.device)
        expr_values = batch["expr_values"].to(self.device)
        rna_target = batch['rna_target'].to(self.device)
        
        cat_protein_embeddings = self.model(
            input_ids=input_ids, 
            expr_values=expr_values, 
            return_cls=True,
        )
        rna_pred = self.dist_head(cat_protein_embeddings)
        loss = F.binary_cross_entropy_with_logits(rna_pred, rna_target.float(), 
                                                  pos_weight=torch.tensor([self.global_pos_weight]).to(self.device))
            
        return loss   
 
 
if __name__ == "__main__":
    import pandas as pd
    from model.model import ProteinLanguageModel
    import anndata as ad
    model = ProteinLanguageModel.load_model_from_checkpoint("checkpoints/basemodel/large-scale-new-10/model_369999.pt")

    DatasetID = '107'
    files = [f"/data/hubowen/protein_data/CLR_data/Dataset{DatasetID}.h5ad"]
    adata = ad.read_h5ad(files[0])
    

    rna_data = ad.read_h5ad(f"/home/hubowen/work/protein_llm/P2RImputation/Dataset{DatasetID}_RNA.h5ad")
    protein_list = adata.var_names.tolist()
    protein_info = pd.read_csv("/home/hubowen/work/protein_llm/protein_info_completed.csv", sep="\t")
    protein2rna_mapping = dict(zip(protein_info["protein_mapping"], protein_info["gene_name"]))
    gene_list = [protein2rna_mapping.get(p, None) for p in protein_list]
    for gene in gene_list:
        if ',' in gene:
            gene_list.remove(gene)
            genes = gene.split(',')
            for g in genes:
                gene_list.append(g)
    rna_data = rna_data[:, rna_data.var.index.isin(gene_list)]
    adata = adata[:adata.n_obs // 2, :]
    print(f"Filtered rna_data to relevant genes:\n{rna_data}")

    batchcorrection_finetuner = P2RImputationTrainer(
        RNA_data=rna_data,
        model=model,
        adata=adata,
        device="cuda",
        lr=1e-3,
        batch_size=512,
        expr_name=f"{DatasetID}_P2R_relevant_genes_halfcell",
        trainable_modules={"lora": 1.0e-3, "reg_proj": 1.0e-5, "expr_head": 5.0e-5}#"lora": 1.0e-3, "reg_proj": 1.0e-5, "expr_head": 5.0e-5},
    )

    # 计算并打印模型参数总量
    total_params_dist_head = sum(p.numel() for p in batchcorrection_finetuner.dist_head.parameters())
    print(f"Total parameters in dist_head: {total_params_dist_head}")
    try:
        print(f"Training files: \n{adata}")
        batchcorrection_finetuner.fit(
            num_epochs=5000,
            save=True,
            save_dir="./checkpoints_P2R_imputation",
            save_epoch_interval=50,
            early_stop=False,
            lora_rank=16,
            lora_alpha=64,
            lora_dropout=0.15,
            show_progress=True,
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    batchcorrection_finetuner.plot_loss_history(save_path=f"./figures/{batchcorrection_finetuner.expr_name}_loss.png")