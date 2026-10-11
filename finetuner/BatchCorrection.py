from anndata import AnnData
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from finetuner.BaseFineTuner import BaseFineTuner, unwrap_model
from model.model import ProteinLanguageModel
import logging
import numpy as np
import ot
from typing import List
import os



'''
该版本尝试了损失计算时对每个批次作为 Target 进行轮询，计算其与其他批次的 OT 损失，并平均化结果。
发现效果并没有提升，现在尝试直接使用原蛋白表达数据进行OT的计算
'''

logger = logging.getLogger(__name__)

def sinkhorn_algorithm(M, epsilon=0.05, max_iter=30):
    """简易 Sinkhorn 算法实现 (PyTorch 版)"""
    with torch.no_grad():
        r = torch.ones(M.shape[0], device=M.device) / M.shape[0]
        c = torch.ones(M.shape[1], device=M.device) / M.shape[1]
        K = torch.exp(-M / epsilon)
        u = torch.ones_like(r)
        for _ in range(max_iter):
            v = c / (torch.matmul(K.t(), u) + 1e-8)
            u = r / (torch.matmul(K, v) + 1e-8)
        pi = u.unsqueeze(1) * K * v.unsqueeze(0)
    return pi

class BatchCorrectionDatset(Dataset):
    """
    Dataset for protein sequences without sorting or EOS appending.
    """
    def __init__(self, 
                 adata: AnnData, 
                 tokenizer, 
                 batch_name: str,
                 celltype_name: str = None,):
        self.adata = adata
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_token = tokenizer.pad_token
        
        if batch_name not in adata.obs_keys():
            raise ValueError(f"batch_name {batch_name} not in adata.obs_keys()")
        batches = adata.obs[batch_name].unique().tolist()
        self.batch2id = {batch: i for i, batch in enumerate(batches)}
        self.batch_name = batch_name
        
        self.celltype_name = celltype_name
        if celltype_name:
            if celltype_name not in adata.obs_keys():
                print(f"celltype_name {celltype_name} not in adata.obs_keys(), using all cosine similarity")
            else:
                cell_types = adata.obs[celltype_name].unique().tolist()
                self.celltype2id = {cell_type: i for i, cell_type in enumerate(cell_types)}
                self.has_label = True
        else:
            self.has_label = False

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        protein_names = list(self.adata.var_names)
        expr_values = self.adata.X[idx].toarray().flatten() if hasattr(self.adata.X[idx], 'toarray') else self.adata.X[idx]
        input_ids = [self.tokenizer.encode(name) for name in protein_names]
        batch_id = self.batch2id[self.adata.obs[self.batch_name].iloc[idx]]  # (1,)
        if self.has_label:
            celltype_id = self.celltype2id[self.adata.obs[self.celltype_name].iloc[idx]]  # (1,)
        else:
            celltype_id = -1  # (1,)
        

        expr_values, input_ids = zip(*[
            (v, i)
            for v, i in zip(expr_values, input_ids)
            if v >= 0
        ])
        expr_values, input_ids = list(expr_values), list(input_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        expr_values = torch.tensor(expr_values, dtype=torch.float)
        batch_id = torch.tensor(batch_id, dtype=torch.long)
        celltype_id = torch.tensor(celltype_id, dtype=torch.long)

        return {
            "input_ids": input_ids,       # (L,)
            "expr_values": expr_values,   # (L,)
            "idx": idx,
            "batch_id": batch_id,         # (1,)
            "celltype_id": celltype_id,   # (1,)
        }
 
class BatchCorrectionFineTuner(BaseFineTuner):
    def __init__(self,
                 batch_name: str,
                 celltype_name: str = None,
                 temperature: float = 0.1,
                 reg: float = 0.02,
                 reg_m: float = 0.5,
                 lambda_intra: float = 0.1,
                 tau_intra: float = 5e2,
                 **kwargs):
        super().__init__(**kwargs)
        self.batch_name = batch_name
        self.celltype_name = celltype_name
        self.temperature = temperature
        self.ref_embeds = None
        self.eval_dataloader = None
        self.reg = reg
        self.reg_m = reg_m
        self.lambda_intra = lambda_intra
        self.tau_intra = tau_intra
               
    def prepare_dataloader(self, **kwargs):
        if self.adata is None:
            raise ValueError("adata is not set")
        
        # 使用改进后的 Dataset
        dataset = BatchCorrectionDatset(
            self.adata, 
            self.tokenizer, 
            batch_name=self.batch_name, 
            celltype_name=self.celltype_name
        )
        self.dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=self.collate_fn,
            drop_last=True # 对比学习通常建议 drop_last 以保证 batch 内有足够的负样本
        )
        
        self.eval_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn
        )
        
    def collate_fn(self, batch):
        pad_token_id =  0 
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids, expr_values, attn_mask = [], [], []
        idx_list, batch_ids, cell_type_ids = [], [], []

        for x in batch:
            L = len(x["input_ids"])
            pad_len = max_len - L

            input_ids.append(F.pad(x["input_ids"], (0, pad_len), value=pad_token_id))
            expr_values.append(F.pad(x["expr_values"], (0, pad_len), value=-1.0))
            attn_mask.append(torch.cat([torch.ones(L), torch.zeros(pad_len)]))
            idx_list.append(x["idx"])
            batch_ids.append(x["batch_id"])
            cell_type_ids.append(x["celltype_id"])

        return {
            "input_ids": torch.stack(input_ids),
            "expr_values": torch.stack(expr_values),
            "attention_mask": torch.stack(attn_mask),
            "idx": idx_list,
            "batch_ids": torch.stack(batch_ids).squeeze(),
            "cell_type_ids": torch.stack(cell_type_ids).squeeze()
        }
   
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
        for name, param in self.model.named_parameters():
            if "norm" in name.lower():
                param.requires_grad = True
 
    def on_fit_start(self, **kwargs):
        self.model.train()
        self.prepare_dataloader()
        
        # 提取固定的全量embedding作为对齐参考
        all_embeds, all_batch_ids = [], []
        with torch.no_grad():
            for batch in self.eval_dataloader:
                z = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    expr_values=batch["expr_values"].to(self.device),
                    return_cls=True
                )
                all_embeds.append(z.cpu().numpy())
                all_batch_ids.append(batch["batch_ids"].numpy())
        
        all_embeds = np.concatenate(all_embeds, axis=0)
        all_batch_ids = np.concatenate(all_batch_ids, axis=0)
        self.ref_embeds = torch.from_numpy(all_embeds).to(self.device)

        if "lora" in self.trainable_modules:         
            lora_rank = kwargs.get("lora_rank", 8)
            lora_alpha = kwargs.get("lora_alpha", 16)
            lora_dropout = kwargs.get("lora_dropout", 0.05)
            target_modules = kwargs.get("target_modules", ["linear1", "linear2", "cls_proj.0", "cls_proj.3"])
            self.deploy_lora(lora_rank, lora_alpha, lora_dropout, target_modules)
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler(T_max=10)
            
    def save_model(self, save_dir: str = None, file_name: str = None, epoch: int = 0, step: int = 0, **kwargs):
        if self.expr_name:
            save_dir = os.path.join(save_dir, self.expr_name)
        os.makedirs(save_dir, exist_ok=True)
        import copy
        core = unwrap_model(self.model)
        core_copy = copy.deepcopy(core)
        
        
        # 在副本上合并LoRA
        core_copy.decoder = core_copy.decoder.merge_and_unload()
        
        # 使用副本的save_model方法保存
        save_path = os.path.join(save_dir, file_name)
        core_copy.save_model(save_path)
        
        # 删除副本释放内存
        del core_copy
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    def early_stopping(self, loss_history, epoch, threshold=1e-3, patience=3, **kwargs) -> bool:
        if len(self.cluster_history) < patience:
            return False

        earlystop = True
        for i in range(patience - 1):
            if self.cluster_history[-patience+i] != self.cluster_history[-patience+i+1]:
                earlystop = False
                break
        
        if earlystop:
            print(
                f"Early stopping at epoch {epoch+1}: "
                f"patience {patience} reached."
            )
            return earlystop
        
        return earlystop

    def compute_intra_loss(self, cell_embed, batch_ids, cell_idx, tau_intra=0.2):
        B = cell_embed.size(0)
        with torch.no_grad():
            same_batch = (batch_ids.unsqueeze(0) == batch_ids.unsqueeze(1))
            z_ref = F.normalize(self.ref_embeds[cell_idx], dim=1)
            sim_ref = torch.matmul(z_ref, z_ref.T)
            self_mask = torch.eye(B, device=self.device).bool()
            intra_mask = same_batch & (~self_mask)
            fill_value = -1e9
            masked_sim_ref = torch.where(intra_mask, sim_ref / tau_intra, torch.full_like(sim_ref, fill_value))
            p_ref = F.softmax(masked_sim_ref / tau_intra, dim=1)
            
        z_curr = F.normalize(cell_embed, dim=1)
        sim_curr = torch.matmul(z_curr, z_curr.T)
        
        masked_sim_curr = torch.where(intra_mask, sim_curr / tau_intra, torch.full_like(sim_curr, fill_value))
        log_p_curr = F.log_softmax(masked_sim_curr, dim=1)
        
        total_intra_kl = F.kl_div(log_p_curr, p_ref, reduction='none')[intra_mask].sum()
        num_active_pairs = intra_mask.sum()
        return total_intra_kl / num_active_pairs if num_active_pairs > 0 else 0.0
  

    def compute_loss(self, batch, **kwargs):
        input_ids = batch["input_ids"].to(self.device)
        expr_values = batch["expr_values"].to(self.device)
        batch_ids = batch["batch_ids"].to(self.device)
        cell_idx = batch["idx"]
        
        # 1. 提取 Embedding
        cell_embed = self.model(
            input_ids=input_ids, 
            expr_values=expr_values, 
            mask=None,
            return_cls=True
        ) # (B, D)

        # 策略：随机选取当前 Batch 中存在的一个 ID 作为 Target，其余作为 Rest
        unique_ids = torch.unique(batch_ids)

        target_id = unique_ids[torch.randint(0, len(unique_ids), (1,))].item()
        
        idx_target = (batch_ids == target_id)
        idx_rest = (batch_ids != target_id)
        
        embed_target = cell_embed[idx_target] # (n_t, D)
        embed_rest = cell_embed[idx_rest]     # (n_r, D)

        # 2. 计算代价矩阵 M (基于余弦相似度)
        norm_t = F.normalize(embed_target, p=2, dim=1)
        norm_r = F.normalize(embed_rest, p=2, dim=1)
        # Cosine Similarity Matrix
        sim_matrix = torch.matmul(norm_t, norm_r.t()) # (n_t, n_r)
        M = 1.0 - sim_matrix

        # 3. 求解 UOT (Unbalanced Optimal Transport)
        with torch.no_grad():
            M_np = M.cpu().numpy().astype(np.float64)
            n_t, n_r = M_np.shape
            # 均匀边际分布
            a, b = np.ones(n_t)/n_t, np.ones(n_r)/n_r
            
            # 使用 UOT：允许质量不平衡，这对“一对多”至关重要
            # 因为 Target 批次中的某种细胞可能只在 Rest 中的某一个子批次出现
            gamma = ot.unbalanced.sinkhorn_knopp_unbalanced(
                a, b, M_np, reg=self.reg, reg_m=self.reg_m
            )
            gamma = torch.from_numpy(gamma).to(self.device).float()
        
        # 归一化 gamma 权重
        gamma_weight = gamma / (gamma.sum(dim=1, keepdim=True) + 1e-8)
        
        pos_logits = sim_matrix / self.temperature # (n_t, n_r)
        
        # 3. 构造批次内负样本得分 (Intra-batch negative scores)
        # 计算 target 批次内部的相互相似度
        norm_t = F.normalize(embed_target, p=2, dim=1)
        intra_sim = torch.matmul(norm_t, norm_t.t()) # (n_t, n_t)
        
        # 屏蔽掉对角线（自己和自己不是负样本）
        mask_self = torch.eye(n_t, device=self.device).bool()
        intra_sim.masked_fill_(mask_self, -1e9) 
        
        n_neg = max(1, int(n_t * 0.25))  # 计算需要选取的负样本数量，至少选 1 个
        
        _, neg_indices = intra_sim.topk(n_neg, dim=1) # (n_t, n_neg)
        
        # 使用 gather 从原始相似度矩阵中提取被选中的负样本 logits
        # intra_neg_logits 维度将变为 (n_t, n_neg)
        intra_neg_logits = torch.gather(intra_sim, 1, neg_indices) / self.tau_intra
        
        combined_logits = torch.cat([pos_logits, intra_neg_logits], dim=1)
        
        # 计算 Softmax 后的对数概率
        log_prob = F.log_softmax(combined_logits, dim=1)
        
        # 只取前 n_r 列（即跨批次部分），用 gamma 权重加权
        # 此时的分母已经包含了同批次的负样本，因此推力更强
        loss_ot = -torch.sum(gamma_weight * log_prob[:, :n_r]) / n_t
        
        loss_intra = self.compute_intra_loss(cell_embed, batch_ids, cell_idx, self.tau_intra)
        
        total_loss = loss_ot + self.lambda_intra * loss_intra
        
        logger.debug(f"Batch OT Loss: {loss_ot.item():.4f}, Intra Loss: {loss_intra:.4f}")

        return total_loss
  




if __name__ == "__main__":
    
    celltype_mapping = {
        # T Cells
        'CD4_Naive': 'CD4 T', 'CD4_EM': 'CD4 T', 'CD4_CM': 'CD4 T', 'CD4_Treg': 'CD4 T', 
        'CD4_Tfh': 'CD4 T', 'CD4_platelet_bind': 'CD4 T', 'CD4': 'CD4 T',
        'CD8_TEMRA': 'CD8 T', 'CD8_CM': 'CD8 T', 'CD8_EM': 'CD8 T', 'CD8_Naive': 'CD8 T', 
        'CD8_TRM': 'CD8 T', 'CD8_proliferating': 'CD8 T', 'CD8': 'CD8 T',
        'MAIT': 'MAIT', 'gdT-Vd2': 'gdT', 'T_Vd2': 'gdT', 'DNT': 'DNT', 'NKT': 'NKT',
        
        # B Cells
        'B_Naive': 'B Cell', 'B_Naive_Intermediate': 'B Cell', 'B_Mem': 'B Cell', 
        'B': 'B Cell', 'Plasmablast': 'Plasmablast',
        
        # Myeloid
        'Mono_Classical': 'Monocyte', 'Mono_NonClassical': 'Monocyte', 'Mono_Intermediate': 'Monocyte',
        'Mono': 'Monocyte', 'Mac-or-Mono': 'Monocyte', 'cDC': 'DC', 'pDC': 'DC', 'Neut': 'Neutrophil',
        
        # NK & Others
        'NK_CD56hiCD16lo': 'NK', 'NK_CD16hi': 'NK', 'NK_proliferating': 'NK', 'NK': 'NK',
        'Platelet': 'Platelet', 'HSPC': 'Progenitor', 'HSC': 'Progenitor', 'cKit+CD3-activated': 'Progenitor',
        'RBC': 'Erythrocyte', 'ILC': 'ILC'
    }

    # 需要剔除或标记为不明的类别
    trash_types = ['dim', 'Mono-T-dblt', 'nan', np.nan]

    def finalize_annotation(adata):
        # 1. 创建新列，默认拷贝细粒度标注
        adata.obs['unified_celltype'] = adata.obs['celltype'].astype(str)
        
        # 2. 如果 celltype 是 'nan'，尝试从 coarsecelltype 补全
        mask_missing = adata.obs['unified_celltype'].isin(['nan', 'None'])
        adata.obs.loc[mask_missing, 'unified_celltype'] = adata.obs.loc[mask_missing, 'coarsecelltype'].astype(str)
        
        # 3. 应用映射字典
        adata.obs['unified_celltype'] = adata.obs['unified_celltype'].map(
            lambda x: celltype_mapping.get(x, 'Unknown')
        )
        
        # 4. 处理特殊项：将原标注中的 dim 或 dblt 设为 Unknown 或直接过滤
        # 这里我们统一设为 'Unknown'
        adata.obs.loc[adata.obs['celltype'].isin(['Mono-T-dblt', 'dim']), 'unified_celltype'] = 'Unknown'
        
        return adata
    
    import anndata as ad
    from model.model import ProteinLanguageModel
    DatasetID="74"
    batch_name = 'Batch'
    
    adata_raw = ad.read_h5ad(f"/data/hubowen/protein_data/CLR_data/Dataset{DatasetID}.h5ad")

    
    model = ProteinLanguageModel.load_model_from_checkpoint("/home/hubowen/work/protein_llm/checkpoints/basemodel/large-scale-new-10/model_369999.pt")
    # model = ProteinLanguageModel.load_model_from_checkpoint("./checkpoints/BatCorr/BatCorr-MultiDataset-kmeans-3/finetune_5.pt")
    
    batchcorrection_finetuner = BatchCorrectionFineTuner(
        model=model,
        adata=adata_raw,
        device="cuda",
        lr=1e-3,
        batch_size=512,
        batch_name=batch_name,
        expr_name=DatasetID,
        trainable_modules={"lora": 1.0e-3, "reg_proj": 1.0e-5, "expr_head": 5.0e-5},
        temperature=0.5,
        reg=0.5,
        reg_m=0.5,
        tau_intra=0.05,
        lambda_intra=10,
    )

    try:
        batchcorrection_finetuner.fit(
            num_epochs=10,
            show_progress=True,
            save=True,
            save_dir="./checkpoints_BatCorr_tmp",
            save_epoch_interval=2,
            early_stop=False,
            lora_rank=16,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=["linear1", "linear2", "cls_proj.0", "cls_proj.3"],
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    batchcorrection_finetuner.plot_loss_history(save_path="./figures/BatCorr_loss.png")