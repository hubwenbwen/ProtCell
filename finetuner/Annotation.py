import torch
from finetuner.BaseFineTuner import BaseFineTuner, unwrap_model
import torch.nn.functional as F
from typing import List, Dict
from torch import nn
import os
from anndata import AnnData
from torch.utils.data import Dataset, DataLoader

'''
26-03-23
该版本实现了以各细胞类型细胞数的根号倒数为权重的加权交叉熵损失函数
相较于无权重版本，稀有类识别更好但是多数类全部被识别为了稀有类
'''



class AnnotationDatset(Dataset):
    """
    Dataset for protein sequences without sorting or EOS appending.
    """
    def __init__(self, 
                 adata: AnnData, 
                 tokenizer, 
                 celltype_name: str,
                 celltype2id: Dict[str, int] = None):
        self.adata = adata
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_token = tokenizer.pad_token
        
        assert celltype_name in adata.obs.columns, f"celltype_name {celltype_name} not found in adata.obs.columns"
        self.celltype_name = celltype_name
        self.celltype2id = celltype2id if celltype2id is not None else {k: i for i, k in enumerate(adata.obs[celltype_name].unique())}
                

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        protein_names = list(self.adata.var_names)
        expr_values = self.adata.X[idx].toarray().flatten() if hasattr(self.adata.X[idx], 'toarray') else self.adata.X[idx]
        input_ids = [self.tokenizer.encode(name) for name in protein_names]
        celltype_id = self.celltype2id[self.adata.obs[self.celltype_name].iloc[idx]]  # (1,)
        

        expr_values, input_ids = zip(*[
            (v, i)
            for v, i in zip(expr_values, input_ids)
            if v >= 0
        ])
        expr_values, input_ids = list(expr_values), list(input_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        expr_values = torch.tensor(expr_values, dtype=torch.float)
        celltype_id = torch.tensor(celltype_id, dtype=torch.long)

        return {
            "input_ids": input_ids,       # (L,)
            "expr_values": expr_values,   # (L,)
            "idx": idx,
            "celltype_id": celltype_id,   # (1,)
        }
 

class DistributionMLP(nn.Module):
    def __init__(self, in_dim: int,d_model: int, out_dim: int = 1, celltype2id: Dict[str, int] = None):
        super(DistributionMLP, self).__init__()
        self.reg_proj = nn.Sequential(
            nn.Linear(in_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.expr_head = nn.Linear(d_model, out_dim)
        
        self.in_dim = in_dim
        self.d_model = d_model
        self.out_dim = out_dim
        self.celltype2id = celltype2id
        
    def forward(self, x):
        x = self.reg_proj(x)
        expr_pred = self.expr_head(x)
        return expr_pred
    
    def save_head(self, save_dir: str, file_name: str):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        # 建议：除了 state_dict，把初始化参数也存进去，方便一键加载
        checkpoint = {
            'state_dict': self.state_dict(),
            'config': {
                'in_dim': self.in_dim,
                'd_model': self.d_model,
                'out_dim': self.out_dim,
                'celltype2id': self.celltype2id
            }
        }
        torch.save(checkpoint, save_path)
        
    @classmethod
    def load_head(cls, path: str, device: str = 'cpu'):
        """
        从文件中直接加载并实例化模型
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
            
        checkpoint = torch.load(path, map_location=device)
        
        # 兼容性检查：判断是只存了 state_dict 还是存了带有 config 的字典
        if 'config' in checkpoint and 'state_dict' in checkpoint:
            # 自动获取保存时的参数
            config = checkpoint['config']
            model = cls(**config) 
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # 如果你之前的模型只保存了 state_dict，
            # 那么这里需要手动传入初始化参数，或者抛出错误
            raise ValueError("此文件仅包含权重，请使用带参数的初始化方式或更新 save_head 函数。")
            
        model.to(device)
        model.eval() # 加载后默认设为评估模式
        return model


class AnnotationTrainer(BaseFineTuner):
    def __init__(self,
                 celltype_name: str,
                 weighted: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        assert celltype_name in self.adata.obs.columns, f"celltype_name {celltype_name} not found in adata.obs.columns"
        self.celltype_name = celltype_name
        self.celltype2id = {k: i for i, k in enumerate(self.adata.obs[celltype_name].unique())}
        self.num_celltypes = len(self.celltype2id)
        self.dist_head = DistributionMLP(in_dim = self.model.decoder.d_model, 
                                         d_model = self.model.decoder.d_model, 
                                         out_dim=self.num_celltypes, 
                                         celltype2id=self.celltype2id).to(self.device)
        self.weighted = weighted
        if self.weighted:
            counts = self.adata.obs[celltype_name].value_counts().to_dict()
            counts = torch.tensor([counts[name] for name in self.celltype2id.keys()], dtype=torch.float)
            self.weights = 1.0 / torch.sqrt(counts)
            self.weights = self.weights / self.weights.sum()
            self.loss = nn.CrossEntropyLoss(weight=self.weights.to(self.device))
        else:
            self.loss = nn.CrossEntropyLoss()
        
    def prepare_dataloader(self, **kwargs):
        dataset = AnnotationDatset(
            self.adata, 
            self.tokenizer, 
            celltype_name=self.celltype_name,
            celltype2id=self.celltype2id
        )
        self.dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=self.collate_fn,
        )
        
    def collate_fn(self, batch):
        pad_token_id =  0 
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids, expr_values = [], []
        idx_list, celltype_id = [], []

        for x in batch:
            L = len(x["input_ids"])
            pad_len = max_len - L

            input_ids.append(F.pad(x["input_ids"], (0, pad_len), value=pad_token_id))
            expr_values.append(F.pad(x["expr_values"], (0, pad_len), value=-1.0))
            idx_list.append(x["idx"])
            celltype_id.append(x["celltype_id"])

        return {
            "input_ids": torch.stack(input_ids),
            "expr_values": torch.stack(expr_values),
            "idx": idx_list,
            "celltype_id": torch.stack(celltype_id).squeeze()
        }
        
    def on_fit_start(self, **kwargs):
        self.model.train()
        self.prepare_dataloader()
        

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
        if self.dist_head is not None:
            self.dist_head.save_head(save_dir, f"head_{epoch}.pt")
            print(f"Saved distribution head to {os.path.join(save_dir, file_name)}")
      
    def compute_loss(self, batch, **kwargs):
        input_ids = batch["input_ids"].to(self.device)
        expr_values = batch["expr_values"].to(self.device)
        celltype_id = batch["celltype_id"].to(self.device)
        
        # 1. 提取 Embedding
        cell_embed = self.model(
            input_ids=input_ids, 
            expr_values=expr_values,
            mask=None,
            return_cls=True
        ) # (B, D)
        
        pred_celltype = self.dist_head(cell_embed) # (B, num_celltypes)
    
        return self.loss(pred_celltype, celltype_id)
 
 
if __name__ == "__main__":
    from preprocess.preprocess import load_and_merge_datasets
    from model.model import ProteinLanguageModel
    model = ProteinLanguageModel.load_model_from_checkpoint("checkpoints/basemodel/large-scale-new-10/model_369999.pt")

    files = ["/data/hubowen/protein_data/CLR_data/Dataset74.h5ad"]
    adata, _ = load_and_merge_datasets(files)
    adata = adata[adata.obs['Batch'].isin(['B4chip1'])].copy()
    print(f"loaded adata from {files}:\n{adata}")
    annotation_finetuner = AnnotationTrainer(
        model=model,
        adata=adata,
        device="cuda",
        lr=1e-3,
        batch_size=512,
        celltype_name='coarsecelltype',
        expr_name="74_B4chip1", # part-proteins, halfcells
    )
    try:
        annotation_finetuner.fit(
            num_epochs=20,
            save=True,
            save_dir="./checkpoints/annotation",
            save_epoch_interval=10,
            early_stop=True,
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    annotation_finetuner.plot_loss_history(save_path="./figures/Imputation_loss.png")