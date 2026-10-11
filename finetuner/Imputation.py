import torch
import torch.nn.functional as F
from finetuner.BaseFineTuner import BaseFineTuner, unwrap_model
from typing import List
from torch import nn
import os

class DistributionHead(nn.Module):
    def __init__(self, embed_dim: int, dist_type: str = 'normal'):
        super().__init__()
        self.dist_type = dist_type
        self.loc_head = nn.Linear(embed_dim, 1)
        self.scale_head = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        loc = self.loc_head(x) # 均值通常不限制范围
        
        if self.dist_type == 'normal':
            # 标准差必须为正，使用 exp 或 softplus
            scale = F.softplus(self.scale_head(x)) + 1e-6 
        elif self.dist_type == 'log_normal':
            loc = torch.clamp(loc, min=-10, max=10) # log_normal 的 loc 其实是 log(mean)，过大或过小都会导致数值不稳定
            scale = F.softplus(self.scale_head(x)) + 1e-6
        elif self.dist_type == 'beta':
            # Beta 分布的 alpha, beta 必须 > 1 (保证单峰) 或 > 0
            scale = F.softplus(self.scale_head(x)) + 1.0 
            loc = F.softplus(loc) + 1.0 # 这里 loc 充当 alpha，scale 充当 beta
        return loc, scale
    
    def save_head(self, save_dir: str, file_name: str):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        torch.save(self.state_dict(), save_path)

class DistributionMLP(nn.Module):
    def __init__(self, in_dim: int,d_model: int, out_dim: int = 1):
        super(DistributionMLP, self).__init__()
        self.reg_proj = nn.Sequential(
            # 1. 第一步：升维。将 256 映射到 1024。
            nn.Linear(in_dim, d_model * 4),
            # nn.LayerNorm(d_model * 4), # 必须在这里稳住分布
            nn.GELU(),
            
            # 2. 第二步：高维特征提取。
            nn.Linear(d_model * 4, d_model * 4),
            # nn.LayerNorm(d_model * 4),
            nn.GELU(),
            
            # 3. 第三步：残差块思路（手动展开版）。
            # 这一步能有效缓解深层网络的“平均化预测”倾向
            nn.Linear(d_model * 4, d_model * 2),
            # nn.LayerNorm(d_model * 2),
            nn.GELU(),
            
            # 4. 最后降维到输出维度
            nn.Linear(d_model * 2, d_model),
            # nn.LayerNorm(d_model)
        )
        self.expr_head = nn.Linear(d_model, out_dim)
        
    def forward(self, x):
        x = self.reg_proj(x)
        expr_pred = self.expr_head(x).squeeze(-1)
        return expr_pred
    
    def save_head(self, save_dir: str, file_name: str):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        torch.save(self.state_dict(), save_path)

class HurdleDistributionMLP(nn.Module):
    def __init__(self, in_dim: int, d_model: int, out_dim: int = 1):
        super(HurdleDistributionMLP, self).__init__()
        
        # 公共特征提取层 (Shared Backbone)
        # 提取蛋白的通用特征，平衡计算量与特征丰富度
        self.reg_proj = nn.Sequential(
            # 1. 第一步：升维。将 256 映射到 1024。
            nn.Linear(in_dim, d_model * 4),
            # nn.LayerNorm(d_model * 4), # 必须在这里稳住分布
            nn.GELU(),
            
            # 2. 第二步：高维特征提取。
            nn.Linear(d_model * 4, d_model * 4),
            # nn.LayerNorm(d_model * 4),
            nn.GELU(),
            
            # 3. 第三步：残差块思路（手动展开版）。
            # 这一步能有效缓解深层网络的“平均化预测”倾向
            nn.Linear(d_model * 4, d_model * 2),
            # nn.LayerNorm(d_model * 2),
            nn.GELU(),
            
            # 4. 最后降维到输出维度
            nn.Linear(d_model * 2, d_model),
            # nn.LayerNorm(d_model)
        )
        
        # 分支 1: 分类头 (Classification Head) - 预测蛋白是否存在 (Gate)
        # 输出 logits，后续配合 BCEWithLogitsLoss 使用
        self.gate_head = nn.Linear(d_model, 1)
        
        # 分支 2: 回归头 (Regression Head) - 预测具体表达量
        # 在末尾增加非负限制
        self.expr_head = nn.Linear(d_model, out_dim)
        nn.init.constant_(self.gate_head.bias, -1.0)
        
    def forward(self, x):
        features = self.reg_proj(x)
        
        # 1. 预测存在的概率 (Logits)
        gate_logits = self.gate_head(features).squeeze(-1)
        
        # 2. 预测潜在的表达强度
        expr_pred = self.expr_head(features).squeeze(-1)
        
        # 在推理阶段，你可以选择结合两者：
        # final_pred = torch.sigmoid(gate_logits) * reg_val
        
        return gate_logits, expr_pred

    def save_head(self, save_dir: str, file_name: str):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        torch.save(self.state_dict(), save_path)

class ImputationTrainer(BaseFineTuner):
    def __init__(self,
                 dist_type=None,
                 train_cls=False, # 选择训练clstoken/蛋白token
                 **kwargs):
        super().__init__(**kwargs)
        self.dist_type = dist_type
        self.dist_head = None
        self.train_cls = train_cls
        if self.dist_type is not None:
            core = unwrap_model(self.model)
            if self.dist_type == "cat_feat":
                self.dist_head = DistributionMLP(in_dim=core.decoder.d_model, d_model=core.decoder.d_model, out_dim=self.adata.n_vars).to(self.device)
            elif self.dist_type == "hurdle_loss":
                self.dist_head = HurdleDistributionMLP(in_dim=core.decoder.d_model, d_model=core.decoder.d_model, out_dim=1).to(self.device)
            else:
                self.dist_head = DistributionHead(core.decoder.d_model, dist_type=dist_type).to(self.device)
        # self.dist_head = nn.Linear(core.decoder.d_model, 1).to(self.device)
        
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
        
        mask = self._generate_mask(expr_values, masked_ratio=self.masked_ratio)

        if isinstance(self.dist_head, DistributionHead):
            protein_embeddings = self.model.get_protein_embedding(
                input_ids=input_ids, 
                expr_values=expr_values, 
                mask=mask,
            )

            loc, scale = self.dist_head(protein_embeddings)
            mask_loc = loc[mask.bool()].squeeze(-1)
            mask_scale = scale[mask.bool()].squeeze(-1)
            mask_expr_values = expr_values[mask.bool()]

            if self.dist_type == 'normal':
                dist = torch.distributions.Normal(mask_loc, mask_scale)
                loss = -dist.log_prob(mask_expr_values).mean()
            elif self.dist_type == 'log_normal':
                eps = 1e-6
                dist = torch.distributions.LogNormal(mask_loc, mask_scale)
                loss = -dist.log_prob(mask_expr_values + eps).mean()
            elif self.dist_type == 'beta':
                dist = torch.distributions.Beta(mask_loc, mask_scale)
                loss = -dist.log_prob(mask_expr_values).mean()
        elif isinstance(self.dist_head, DistributionMLP):
            cat_protein_embeddings = self.model(
                input_ids=input_ids, 
                expr_values=expr_values, 
                return_cls=True,
                mask=mask,
            )
            expr_preds = self.dist_head(cat_protein_embeddings)
            loss = F.mse_loss(expr_preds[mask.bool()], expr_values[mask.bool()])
                
        elif isinstance(self.dist_head, HurdleDistributionMLP):
            cat_protein_embeddings = self.model.get_protein_embedding(
                input_ids=input_ids, 
                expr_values=expr_values, 
                mask=mask,
            )
            gate_logits, reg_val = self.dist_head(cat_protein_embeddings)
            mask_gate_logits = gate_logits[mask.bool()]
            mask_reg_val = reg_val[mask.bool()]
            mask_expr_values = expr_values[mask.bool()]

            # 分类损失：预测蛋白是否存在
            binary_labels = (mask_expr_values > 0).float()
            # 计算正样本权重以应对稀疏性
            pos_weight = (binary_labels == 0).sum() / (binary_labels == 1).sum().clamp(min=1)

            # 分类损失：增加 pos_weight
            classification_loss = F.binary_cross_entropy_with_logits(
                mask_gate_logits, 
                binary_labels, 
                pos_weight=pos_weight
            )

            # 确保回归损失不会因为样本太少而导致 loss 异常
            if binary_labels.sum() > 0:
                # 仅针对 Ground Truth > 0 的点计算回归
                regression_loss = F.mse_loss(
                    mask_reg_val[binary_labels.bool()], 
                    mask_expr_values[binary_labels.bool()]
                )
            else:
                regression_loss = 0.0

            # 建议在 Log 中分开记录两个 Loss，观察哪个更难降下来
            loss = classification_loss + 1.0 * regression_loss
            logger.info(f"Classification Loss: {classification_loss.item():.4f}, Regression Loss: {regression_loss if isinstance(regression_loss, float) else regression_loss.item():.4f}")
        else:
            expr_preds, global_pred = self.model(input_ids=input_ids, expr_values=expr_values, mask=mask)
            if self.train_cls:
                selected_pred = torch.gather(global_pred, 1, input_ids)
                loss = F.mse_loss(selected_pred[mask], expr_values[mask])
            else:
                mask_expr_values = expr_values[mask.bool()]
                squared_errors = (expr_preds - mask_expr_values) ** 2
                # weights = torch.log1p(mask_expr_values) + 1.0
                
                # loss = (squared_errors * weights).mean()
                loss = squared_errors.mean()
            
        return loss   
 
 
if __name__ == "__main__":
    from model.model import ProteinLanguageModel
    import anndata as ad
    model = ProteinLanguageModel.load_model_from_checkpoint("checkpoints/basemodel/large-scale-new-10/model_369999.pt")

    DatasetID = '83'
    files = [f"/data/hubowen/protein_data/CLR_data/Dataset{DatasetID}.h5ad"]
    adata = ad.read_h5ad(files[0])
    print(f"loaded adata from {files}:\n{adata}")
    adata = adata[adata.obs['batch'].isin(['1'])].copy() # half cells for training, half cells for testing
    # adata = adata[:adata.n_obs // 2, :].copy()

    batchcorrection_finetuner = ImputationTrainer(
        model=model,
        train_cls=True,
        adata=adata,
        device="cuda",
        lr=1e-3,
        batch_size=512,
        masked_ratio=0.15,
        expr_name=f"{DatasetID}_batch1", # part-proteins, halfcells
        dist_type=None,
        trainable_modules={"lora": 1.0e-3, "reg_proj": 1.0e-5, "expr_head": 5.0e-5}#"lora": 1.0e-3, "reg_proj": 1.0e-5, "expr_head": 5.0e-5},
    )
    try:
        batchcorrection_finetuner.fit(
            num_epochs=50,
            save=True,
            save_dir="./checkpoints_imputation",
            save_epoch_interval=10,
            early_stop=True,
            lora_rank=16,
            lora_alpha=64,
            lora_dropout=0.15,
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    batchcorrection_finetuner.plot_loss_history(save_path="./figures/Imputation_loss.png")