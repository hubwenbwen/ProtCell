import torch
from torch.utils.data import DataLoader
from anndata import AnnData
from trainer.mask_trainer import ProteinSequenceDataset
import os
from typing import Union, Callable, List, Dict
import random
import torch.nn.functional as F
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                       filename="fine_tuner.log",
                       )

def unwrap_model(model):
    if isinstance(model, (torch.nn.parallel.DistributedDataParallel, 
                         torch.nn.parallel.DataParallel)):
        return model.module
    return model

def diff_report(m1, m2, eps=1e-8):
    reports = []
    for (n, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        d = (p1.detach() - p2.detach()).abs()
        max_diff = d.max().item()
        mean_diff = d.mean().item()
        if max_diff > eps:
            reports.append((n, max_diff, mean_diff))
    return reports

def test_lora(model):
    for n, p in model.named_parameters():
        if "lora" in n:
            print(n, p.grad is None, p.grad.abs().mean() if p.grad is not None else None)

class BaseFineTuner():
    def __init__(self, 
                 model: torch.nn.Module, 
                 adata: Union[AnnData, List[str]],
                 device: str = "cuda",
                 lr: float = 1e-3,
                 trainable_modules: Dict[str, float] = {"lora": 1.0e-3, "reg_proj": 1.0e-5, "expr_head": 1.0e-5},
                 optimizer_cls: Union[str, Callable] = 'adamw',
                 scheduler_cls: Union[str, Callable] = 'cosine',
                 warmup_steps: int = 0,
                 batch_size: int = 32,
                 masked_ratio: float = 0.15,
                 weight_decay: float = 1e-2,
                 expr_name: str = "test",
                 seed: int = 42,
                 **kwargs
                 ):
        self.model = model.to(device)
        self.tokenizer = model.tokenizer
        self.device = device
        self.save_dir = "./checkpoints_finetune"
        self.lr = lr
        
        self.batch_size = batch_size
        self.dataloader = None
        self.masked_ratio = masked_ratio
        self.weight_decay = weight_decay
        self.expr_name = expr_name
        self.loss_history = []
        self._no_improve_count = 0
        
        self.optimizer_cls = optimizer_cls
        self.scheduler_cls = scheduler_cls
        self.warmup_steps = warmup_steps
        self.trainable_modules = trainable_modules
        self.optimizer = None
        self.scheduler = None
        # self.optimizer = self._get_optimizer(optimizer_cls)
        # self.scheduler = self._get_scheduler(scheduler_cls, warmup_steps, T_max=10)
        
        self.adata = None
        self.vars_names = None
        if isinstance(adata, List):
            self._prepare_data(adata)
        elif isinstance(adata, AnnData):
            self.adata = adata
            self.vars_names = adata.var_names.tolist()
        else:
            raise ValueError("Unsupported data type for new_data")
        
        self.seed = seed
        self.set_seed(self.seed)
  
    def _prepare_data(self, new_files):
        from preprocess.preprocess import load_dataset
        self.adata, _, self.vars_names = load_dataset(new_files, test_size=0.0)
        
    def collate_fn(self, batch):
        pad_token_id =  0 
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids, expr_values, attn_mask, idx_list = [], [], [], []

        for x in batch:
            L = len(x["input_ids"])
            pad_len = max_len - L

            input_ids.append(F.pad(x["input_ids"], (0, pad_len), value=pad_token_id))
            expr_values.append(F.pad(x["expr_values"], (0, pad_len), value=-1.0))
            attn_mask.append(torch.cat([torch.ones(L), torch.zeros(pad_len)]))
            idx_list.append(x["idx"])

        return {
            "input_ids": torch.stack(input_ids),
            "expr_values": torch.stack(expr_values),
            "attention_mask": torch.stack(attn_mask),
            "idx": idx_list
        }
        
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

        if not parameter_groups:
            raise ValueError("No trainable parameters found. Check requires_grad or trainable_modules.")

        opt_name = self.optimizer_cls.lower() if isinstance(self.optimizer_cls, str) else ""
        
        if opt_name == 'adamw':
            return torch.optim.AdamW(parameter_groups, weight_decay=self.weight_decay)
        elif opt_name == 'adam':
            return torch.optim.Adam(parameter_groups, weight_decay=self.weight_decay)

        return self.optimizer_cls(parameter_groups)

    def _get_scheduler(self, **scheduler_kwargs):
        assert self.optimizer is not None, "optimizer is not set"
        
        if isinstance(self.scheduler_cls, str):
            if self.scheduler_cls.lower() == 'cosine':
                main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **scheduler_kwargs)
            elif self.scheduler_cls.lower() == 'step':
                main_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **scheduler_kwargs)
            elif self.scheduler_cls.lower() == 'linear':
                main_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 - epoch / scheduler_kwargs.get("t_max", 10))
            else:
                raise ValueError(f"Unsupported scheduler: {self.scheduler_cls}")
            
            if self.warmup_steps > 0:
                # LambdaLR 的 warmup wrapper
                def lr_lambda_warmup(current_step):
                    if current_step < self.warmup_steps:
                        # linear warmup from 0 → 1
                        return float(current_step) / float(max(1, self.warmup_steps))
                    # use the main scheduler’s lambda if available
                    if isinstance(main_scheduler, torch.optim.lr_scheduler.LambdaLR):
                        return main_scheduler.lr_lambdas[0](current_step - self.warmup_steps)
                    return 1.0  # default (no further decay)

                warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=lr_lambda_warmup
                )
                # 组合两个 scheduler：先 warmup，再接主 scheduler
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[self.warmup_steps]
                )
                return scheduler
            else:
                return main_scheduler
            
        else:
            return self.scheduler_cls(self.optimizer, **scheduler_kwargs)

    @torch.no_grad()
    def _generate_mask(self, expr_values: torch.Tensor, masked_ratio: float = None):
        if masked_ratio is None:
            masked_ratio = self.masked_ratio

        if not (0 <= masked_ratio <= 1):
            raise ValueError("masked_ratio must be in [0, 1]")

        B, L = expr_values.shape
        device = expr_values.device
        mask = torch.zeros((B, L), dtype=torch.bool, device=device)

        for b in range(B):
            nonpad_idx = torch.where(expr_values[b] >= 0)[0]  # tensor, 在同一 device
            nz_len = nonpad_idx.size(0)
            if nz_len == 0:
                continue

            n_mask = int(nz_len * masked_ratio)
            # 防止 n_mask 超过 nz_len（保险起见）
            n_mask = min(n_mask, nz_len)

            # 在同一 device 上随机选择 n_mask 个不同位置
            perm = torch.randperm(nz_len, device=device)[:n_mask]
            chosen = nonpad_idx[perm]  # 选择实际的列索引
            mask[b, chosen] = True

        return mask
    
    def save_model(self,
                   save_dir: str = None, 
                   file_name: str = None,
                   epoch: int = 0,
                   step: int = 0,):
        save_dir = save_dir if save_dir is not None else self.save_dir
        file_name = file_name if file_name is not None else f"finetune_{epoch+1}.pt"
        
        if self.expr_name:
            save_dir = os.path.join(save_dir, self.expr_name)
        
        os.makedirs(save_dir, exist_ok=True)
        unwrap_model(self.model).save_model(os.path.join(save_dir, file_name))
       
    def set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)   
       
    def prepare_dataloader(self, **kwargs):
        if self.adata is None:
            raise ValueError("adata is not set for dataloader preparation")
        dataset = ProteinSequenceDataset(self.adata, self.tokenizer)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        
    def on_fit_start(self, **kwargs):
        self.model.train()
        self.prepare_dataloader(**kwargs)
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler(T_max=10)
    
    def early_stopping(self, loss_history, epoch, threshold=1e-3, patience=3, **kwargs) -> bool:
        if len(loss_history) < 2:
            return False

        rel_change = abs(loss_history[-1] - loss_history[-2]) / (loss_history[-2] + 1e-8)

        # 如果变化小于阈值 → 触发一次“不再显著改进”
        if rel_change < threshold:
            self._no_improve_count += 1
        else:
            self._no_improve_count = 0

        # 达到 patience 次后停止
        if self._no_improve_count >= patience:
            print(
                f"Early stopping at epoch {epoch+1}: "
                f"loss change {rel_change:.6f} < {threshold}, "
                f"patience {patience} reached."
            )
            return True
        
        return False

    def fit(self,
            num_epochs: int = 5,
            save: bool = True,
            save_dir: str = "./checkpoints_finetune",
            save_epoch_interval: int = 1,
            start_epoch: int = 0,
            early_stop: bool = True,
            threshold: float = 1e-3,
            patience: int = 3,
            show_progress: bool = False,
            **kwargs):
        """
        Fine-tune model when new proteins are added.
        - new_data: AnnData, ProteinSequenceDataset, or DataLoader
        - num_epochs: number of epochs for fine-tuning
        - lr: learning rate for fine-tuning
        - save: whether to save checkpoints
        """
        

        self.on_fit_start(**kwargs)
        self.model.to(self.device)
        self.model.train()
        self.loss_history.clear()
        for epoch in range(start_epoch, num_epochs):
            total_loss = self.train_epoch(epoch, show_progress, **kwargs)
            self.scheduler.step()

            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(self.dataloader):.4g}")
            self.loss_history.append(total_loss / len(self.dataloader))
            
            # Early stopping based on loss change
            if epoch > 0 and early_stop:
                if self.early_stopping(self.loss_history, epoch, threshold=threshold, patience=patience, **kwargs):
                    break
            
            if save and (epoch + 1) % save_epoch_interval == 0:
                self.save_model(save_dir=save_dir, file_name=f"finetune_{epoch+1}.pt", epoch=epoch+1, step=0)
        if save:
            self.save_model(save_dir=save_dir, file_name=f"finetune_{num_epochs}.pt", epoch=num_epochs, step=0)

    def train_epoch(self, epoch, show_progress, **kwargs):
        total_loss = 0.0
        dataloader = tqdm(self.dataloader, desc=f"Epoch {epoch+1}") if show_progress else self.dataloader
        for batch in dataloader:
            loss = self.compute_loss(batch, **kwargs)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            
        return total_loss
    
    def compute_loss(self, batch, **kwargs):
        input_ids = batch["input_ids"].to(self.device)
        expr_values = batch["expr_values"].to(self.device)
        attn_mask = batch["attention_mask"].to(self.device)

        mask = self._generate_mask(expr_values, masked_ratio=self.masked_ratio)

        expr_preds, global_pred = self.model(input_ids=input_ids, expr_values=expr_values, mask=mask)
        mask_expr_values = expr_values[mask.bool()]
        loss_expr = (F.mse_loss(expr_preds, mask_expr_values)).sum() / mask.sum()
        
        loss_global = 0.0
        if global_pred is not None:
            selected_pred = torch.gather(global_pred, 1, input_ids)  # (B, L)
            loss_global = (F.mse_loss(selected_pred[attn_mask.bool()], expr_values[attn_mask.bool()])).sum() / attn_mask.sum()

        loss = loss_expr + loss_global
        return loss

    def plot_loss_history(self, save_path: str = None):
        import matplotlib.pyplot as plt
        plt.plot(self.loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()    

 
            
if __name__ == '__main__':
    pass
    

    
    