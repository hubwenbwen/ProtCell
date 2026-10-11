# trainer.py

from math import exp, ceil
from unittest import result
from sympy import false
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from anndata import AnnData
from typing import List, Dict, Union, Callable
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

from preprocess.preprocess import load_dataset
from scipy.stats import pearsonr
from model.model import ProteinLanguageModel
from databank.DataBankReader import DataBankReader
from databank.DataBankDataset import DataBankDataset, RankShardDataset

import os
import logging
from tqdm import tqdm
from datetime import datetime

import torch.distributed as dist

logger = logging.getLogger(__name__)


class ProteinSequenceDataset(Dataset):
    """
    Dataset for protein sequences without sorting or EOS appending.
    """
    def __init__(self, adata: AnnData, tokenizer):
        self.adata = adata
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_token = tokenizer.pad_token

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
            "idx": idx
        }

def collate_fn(batch):
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

def unwrap_model(model):
    if isinstance(model, (torch.nn.parallel.DistributedDataParallel, 
                         torch.nn.parallel.DataParallel)):
        return model.module
    return model

class MaskTrainer:
    def __init__(self, 
                model: nn.Module,
                device: str='cuda',
                batch_size: int=2048,
                lr: float=1e-5,
                weight_decay: float=0.01,
                log_epoch: int=1,
                log_steps: int=10,
                gradient_accumulation_steps: int=32,
                optimizer_cls: Union[str, Callable] = 'adamw',
                scheduler_cls: Union[str, Callable] = 'cosine',
                warmup_ratio: int = 0,
                num_of_all_samples: int = 0,
                scheduler_kwargs: dict = None,
                masked_ratio: float=0.15,
                expr_name: str = None,
                loss_type: str="mse",
                precision: str="float32",
                ):

        self.model = model
        self.tokenizer = model.tokenizer
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.masked_ratio = masked_ratio
        self.log_epoch = log_epoch
        self.log_steps = log_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.optimizer_cls = optimizer_cls
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.warmup_steps = warmup_ratio * (num_of_all_samples // batch_size)

        self.model = self.model.to(self.device)
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.writer = None
        self.save_dir = None
        self.expr_name = expr_name
        
        self.start_epoch = 0
        self.start_steps = 0
        
        if precision == "fp16":
            self.autocast_dtype = torch.float16
            self.scaler = torch.amp.GradScaler()
        elif precision == "bf16":
            self.autocast_dtype = torch.bfloat16
            self.scaler = None
        elif precision == "float32":
            self.autocast_dtype = torch.float32
            self.scaler = None
        else:
            raise ValueError("Unsupported precision. Please use fp16, bf16 or float32")
        
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Current CUDA device does not support bfloat16 precision")

        
        if loss_type == "mse":
            self.loss = nn.MSELoss(reduction='none')
        elif loss_type == "smooth_l1":
            self.loss = nn.SmoothL1Loss(reduction='none')
        elif loss_type == "huber":
            self.loss = nn.HuberLoss(delta=1.0)
        
    def _get_optimizer(self):
        params = params = self.model.parameters()

        if isinstance(self.optimizer_cls, str):
            if self.optimizer_cls.lower() == 'adamw':
                return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
            elif self.optimizer_cls.lower() == 'adam':
                return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
            elif self.optimizer_cls.lower() == 'sgd':
                return torch.optim.SGD(params, lr=self.lr, momentum=0.9)
            else:
                raise ValueError(f"Unsupported optimizer: {self.optimizer_cls}")
        else:
            return self.optimizer_cls(params, lr=self.lr)

    def _get_scheduler(self):
        if isinstance(self.scheduler_cls, str):
            if self.scheduler_cls.lower() == 'cosine':
                main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **self.scheduler_kwargs)
            elif self.scheduler_cls.lower() == 'step':
                main_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **self.scheduler_kwargs)
            elif self.scheduler_cls.lower() == 'linear':
                main_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 - epoch / self.scheduler_kwargs.get("t_max", 10))
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
            return self.scheduler_cls(self.optimizer, **self.scheduler_kwargs)

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

    def _variance_regularizer(self, z, eps=1e-4, min_std=0.05):
        # z: (B, D)
        std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)   # per-dim std (tensor length D)
        # 惩罚那些 std < min_std
        loss_var = torch.mean(F.relu(min_std - std))
        return loss_var
    
    @staticmethod
    def _vicreg_loss(z1: torch.Tensor,
                     z2: torch.Tensor,
                     sim_coeff: float = 25.0,
                     var_coeff: float = 25.0,
                     cov_coeff: float = 1.0,
                     eps: float = 1e-4,
                     var_eps: float = 1e-4,
                     gamma: float = 1.0):
        """
        Compute VICReg loss between two embeddings z1, z2 (B, D).

        Returns three scalars: (total_loss, sim_loss, var_loss + cov_loss sum)
        but here we return total loss and components for logging.
        """
        assert z1.ndim == 2 and z2.ndim == 2, "z must be (B, D)"
        B, D = z1.shape

        # ---- 1) invariance (MSE) ----
        sim_loss = F.mse_loss(z1, z2)  # mean over all elements

        # ---- 2) variance term ----
        # per-dim std (use unbiased=False for stability with small batches)
        std_z1 = torch.sqrt(z1.var(dim=0, unbiased=False) + var_eps)
        std_z2 = torch.sqrt(z2.var(dim=0, unbiased=False) + var_eps)
        # penalize dims with std < gamma
        var_loss = torch.mean(F.relu(gamma - std_z1)) + torch.mean(F.relu(gamma - std_z2))

        # ---- 3) covariance term ----
        def covariance_loss(z):
            # z: (B, D) center the features
            z = z - z.mean(dim=0, keepdim=True)
            # compute covariance matrix (D x D)
            # use (z^T z) / (B - 1) for unbiased estimate; if B==1 this will be zero/NaN -> avoid B==1
            cov = (z.T @ z) / (B - 1 if B > 1 else 1.0)
            # zero out diagonal
            diag = torch.diag(torch.diag(cov))
            off_diag = cov - diag
            # square and sum, normalize by D
            return (off_diag ** 2).sum() / D

        cov_loss = covariance_loss(z1) + covariance_loss(z2)

        total = sim_coeff * sim_loss + var_coeff * var_loss + cov_coeff * cov_loss
        return total, sim_loss.detach(), var_loss.detach(), cov_loss.detach()

    def train(self, 
              train_data,
              test_data=None,
              num_epochs: int=10, 
              save: bool=false,
              save_dir: str='./checkpoints',
              save_epoch_interval: int=10,
              save_steps_interval: int=2000,
              log_dir: str='./logs',
              distributed: bool = False,
              local_rank: int = 0,
              num_workers: int = 0,
              seed: int = 42,
                ):
        
        if distributed:
            # 初始化 DDP 环境
            torch.cuda.set_device(local_rank)
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            device = torch.device(f"cuda:{local_rank}")
            is_main_process = dist.get_rank() == 0
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            device = self.device
            is_main_process = True
            world_size = 1
            rank = 0
        
        # ===== TensorBoard SummaryWriter ===== #
        if is_main_process:
            if self.expr_name is None:
                log_dir = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            else:
                log_dir = os.path.join(log_dir, self.expr_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            self.writer = SummaryWriter(log_dir=log_dir)
                    
        # ====== 模型分发 ======
        self.model.to(device)
        if distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
            )
                
        # ====== DataLoader ======
        uses_databank = None
        global_dataset = None
        if isinstance(train_data, DataLoader):
            dataloader = train_data
        elif isinstance(train_data, DataBankDataset):
            global_dataset = train_data
            uses_databank = True
        else:
            if isinstance(train_data, AnnData):
                dataset = ProteinSequenceDataset(train_data, self.tokenizer)
            elif isinstance(train_data, ProteinSequenceDataset):
                dataset = train_data
            
            else:
                raise ValueError("Unsupported data type for train_data")
            
            if distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset, shuffle=True
                )
                shuffle = False
            else:
                train_sampler = None
                shuffle = True

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory= num_workers > 0,
                collate_fn=collate_fn,
            )
        
        # ====== 测试集 DataLoader ======
        if test_data is not None:
            if isinstance(test_data, DataLoader):
                test_dataloader = test_data
            else:
                if isinstance(test_data, AnnData):
                    test_dataset = ProteinSequenceDataset(test_data, self.tokenizer)
                elif isinstance(test_data, ProteinSequenceDataset):
                    test_dataset = test_data
                elif isinstance(test_data, DataBankReader):
                    test_dataset = DataBankDataset(test_data)
                elif isinstance(test_data, DataBankDataset):
                    test_dataset = test_data
                else:
                    raise ValueError("Unsupported data type for test_data")
                
                test_dataloader = DataLoader(
                    test_dataset, 
                    batch_size=self.batch_size, 
                    shuffle=False, 
                    num_workers=num_workers,
                    pin_memory = num_workers > 0,
                    collate_fn=collate_fn
                )
        else:
            test_dataloader = None
        
        self.model.train()
        core = unwrap_model(self.model)
        
        try:
            for epoch in range(self.start_epoch, num_epochs):
                if uses_databank:
                    rank_dataset, min_samples = global_dataset.get_rank_dataset(rank, world_size, epoch, cache_size=global_dataset.cache_size, device=device, distributed=distributed)
                    g = torch.Generator()
                    g.manual_seed(seed + epoch)
                    sampler = torch.utils.data.RandomSampler(rank_dataset, generator=g)
                    dataloader = DataLoader(
                        rank_dataset,
                        batch_size=self.batch_size,
                        sampler=sampler,
                        num_workers=num_workers,
                        pin_memory = num_workers > 0,
                        persistent_workers = (num_workers > 0),
                        collate_fn = collate_fn,
                    )
                    max_steps = ceil(min_samples / self.batch_size)
                    if distributed:
                        dist.barrier(device_ids=[local_rank])
                
                else:
                    if distributed and hasattr(dataloader, 'sampler') and isinstance(
                        dataloader.sampler, torch.utils.data.distributed.DistributedSampler
                    ):
                        dataloader.sampler.set_epoch(epoch)
                        max_steps = len(dataloader)
                
                epoch_loss = 0.0
                epoch_steps = 0
                
                # print(f"[Rank {dist.get_rank()}] max_steps: {max_steps}, dataloader length: {len(dataloader)}, start steps: {self.start_steps}")
                
                for step, batch in tqdm(enumerate(dataloader), total=max_steps, desc=f"Epoch {epoch+1}", leave=False, disable=not is_main_process):
                    if step <= self.start_steps:
                        continue
                    else:
                        self.start_steps = 0 
                    if step >= max_steps:
                        break
                    input_ids = batch["input_ids"].to(device)          # (B, L)
                    expr_values = batch["expr_values"].to(device)      # (B, L)
                    attn_mask = batch["attention_mask"].to(device)     # (B, L)
                    idx = batch["idx"]
                    
                    with torch.autocast(device_type='cuda', dtype=self.autocast_dtype):
                        # === Generate mask and predict ===
                        mask = self._generate_mask(expr_values, masked_ratio=self.masked_ratio)
                        expr_preds, global_pred = self.model(input_ids=input_ids, expr_values=expr_values, mask=mask)
                        # === Compute losses ===
                        mask_expr_values = expr_values[mask.bool()]
                        loss_expr = (self.loss(expr_preds, mask_expr_values)).sum() / mask.sum()

                        # === Compute Cls Loss ===
                        loss_global = 0.0
                        if global_pred is not None:
                            selected_pred = torch.gather(global_pred, 1, input_ids)  # (B, L)
                            loss_global = (self.loss(selected_pred[attn_mask.bool()], expr_values[attn_mask.bool()])).sum() / attn_mask.sum()
                            
                        loss = loss_expr + loss_global

                    # proceed with backward, gradient accumulation etc.
                    loss = loss / self.gradient_accumulation_steps
                    
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    epoch_loss += loss.item() * self.gradient_accumulation_steps
                    epoch_steps += 1

                    # # logging for nan and inf gradients
                    # for name, param in self.model.named_parameters():
                    #     if param.grad is not None:
                    #         assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN or Inf"

                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.scheduler.step()
                    
                    global_step = epoch * len(dataloader) + step
                    
                    if step % self.log_steps == 0 and is_main_process:
                        self.writer.add_scalar("train/loss", loss.item() * self.gradient_accumulation_steps, global_step)
                        self.writer.add_scalar("train/loss_expr", loss_expr.item(), global_step)
                        if 'loss_global' in locals() and loss_global != 0.0:
                            self.writer.add_scalar("train/loss_global", loss_global.item(), global_step)
                    
                    if save and (global_step + 1) % save_steps_interval == 0:
                        if is_main_process:
                            self.save_model(
                                epoch=epoch,
                                step=step,
                                global_step=epoch * len(dataloader) + step,
                                save_trainer=True,
                                print_info=True,
                            )
                        # if distributed:
                        #     dist.barrier(device_ids=[local_rank])

                # ===== after one epoch ==== #
                # print(f"[Rank {dist.get_rank()}] finish epoch {epoch+1}, loss: {epoch_loss/epoch_steps:.4f}")
                # if (epoch + 1) % self.log_epoch == 0 and is_main_process:
                #     # logger.info(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f}")
                #     if test_dataloader is not None:
                #         results_pred, results_global = self.evaluate(test_dataloader, epsilon=0.1, return_var_ratio=True, show_scatter=False)
                #         def results_writer(results, prefix="eval"):
                #             self.writer.add_scalar(f"{prefix}/expr_mae", results['expr_mae'], epoch)
                #             self.writer.add_scalar(f"{prefix}/expr_rmse", results['expr_rmse'], epoch)
                #             self.writer.add_scalar(f"{prefix}/expr_smape", results['expr_smape'], epoch)
                #             self.writer.add_scalar(f"{prefix}/expr_pearsonr", results['expr_pearsonr'], epoch)
                #             self.writer.add_scalar(f"{prefix}/expr_r2", results['expr_r2'], epoch)
                #             self.writer.add_scalar(f"{prefix}/accuracy", results['accuracy'], epoch)
                #             self.writer.add_scalar(f"{prefix}/cell_embedding_var_ratio", results['cell_embedding_var_ratio'], epoch)
                #             self.writer.add_scalars(f"{prefix}/mean",{"pred": results['pred_mean'], "true": results['true_mean']}, epoch)
                #             self.writer.add_scalars(f"{prefix}/std",{"pred": results['pred_std'], "true": results['true_std']}, epoch)
                #         results_writer(results_pred, prefix="eval/pred")
                #         if results_global is not None:
                #             results_writer(results_global, prefix="eval/global")
                        
                #     # ===== VICReg 收敛检测 =====
                #     if vicreg_enabled:
                #         vicreg_history.append(results_pred['cell_embedding_var_ratio'])
                #         if len(vicreg_history) > vicreg_patience:
                #             diffs = [abs(vicreg_history[i] - vicreg_history[i-1]) for i in range(1, len(vicreg_history))]
                #             if max(diffs[-vicreg_patience:]) < vicreg_tol:
                #                 vicreg_enabled = False
                #                 print(f"VICReg loss converged, disabling it from epoch {epoch+1}")
                            
                # # 同步 vicreg_enabled 状态到所有进程
                # if distributed:
                #     vicreg_tensor = torch.tensor([1 if vicreg_enabled else 0], 
                #                             dtype=torch.int, device=device)
                #     dist.broadcast(vicreg_tensor, src=0)
                #     vicreg_enabled = bool(vicreg_tensor.item())
                #     dist.barrier(device_ids=[local_rank])
                
                # save model
                # if save and (epoch + 1) % save_epoch_interval == 0 and is_main_process:
                #     self.save_model(
                #         epoch=epoch,
                #         step=step,
                #         global_step=epoch * len(dataloader) + step,
                #         save_trainer=True,
                #         print_info=True,
                #     )
        
        finally:
            # 清理
            if distributed:
                try:
                    dist.barrier(device_ids=[local_rank])
                except Exception:
                    pass
                # 确保进程组已初始化再尝试销毁
                
            if is_main_process:
                try:
                    print(f"[Rank 0] Saving emergency checkpoint at epoch={epoch}, step={step}")
                    self.save_model(
                        epoch=epoch,
                        step=step,
                        global_step=epoch * len(dataloader) + step,
                        save_trainer=True,
                        print_info=True,
                    )
                except Exception as e:
                    print(f"[Rank 0] Failed to save checkpoint in finally block: {e}")
                
            if distributed and dist.is_initialized():
                try:
                    dist.barrier(device_ids=[local_rank])
                except Exception:
                    pass
                
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass
            if is_main_process:
                try:
                    self.writer.close()
                except Exception:
                    pass

    @torch.no_grad()
    def evaluate(self, 
                  eval_data, 
                  eval_batch_size: int = None,
                  epsilon: float = 0.1,
                  return_var_ratio: bool = True,
                  show_scatter: bool = False
                  ) -> Dict[str, float]:
        self.model.eval()
        self.model.to(self.device)

        all_expr_preds = []
        all_global_preds = []
        all_expr_labels = []
        
        all_cell_embeddings = []
        core = unwrap_model(self.model)  # 访问真实模型对象

        if eval_batch_size is None:
            eval_batch_size = self.batch_size

        if isinstance(eval_data, AnnData):
            dataset = ProteinSequenceDataset(eval_data, self.tokenizer)
            eval_loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn)
        elif isinstance(eval_data, ProteinSequenceDataset):
            eval_loader = DataLoader(eval_data, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn)
        elif isinstance(eval_data, DataLoader):
            eval_loader = eval_data
        elif isinstance(eval_data, DataBankReader):
            dataset = DataBankDataset(eval_data)
            eval_loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn)
        elif isinstance(eval_data, DataBankDataset):
            eval_loader = DataLoader(eval_data, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn)
        else:
            raise ValueError("Unsupported data type for eval_data")

        def evaluate_metrics(all_expr_preds, all_expr_labels, all_cell_embeddings=None, return_var_ratio=return_var_ratio, show_scatter=show_scatter, name=""):    
            # ====== expr evaluation ======
            expr_preds_all = torch.cat(all_expr_preds)
            expr_labels_all = torch.cat(all_expr_labels)

            mae = torch.mean(torch.abs(expr_preds_all - expr_labels_all)).item()
            rmse = torch.sqrt(torch.mean((expr_preds_all - expr_labels_all) ** 2)).item()
            smape = (2 * torch.abs(expr_preds_all - expr_labels_all) / (torch.abs(expr_preds_all) + torch.abs(expr_labels_all) + 1e-6)).mean().item()
            # ====== Accuracy ======
            correct = torch.abs(expr_preds_all - expr_labels_all) < epsilon
            accuracy = correct.float().mean().item()
            if show_scatter:
                max_points = 10000
                if len(expr_labels_all) > max_points:
                    idx = np.random.choice(len(expr_labels_all), max_points, replace=False)
                    y_true, y_pred = expr_labels_all[idx], expr_preds_all[idx]
                plt.figure(figsize=(5,5))
                plt.scatter(y_true, y_pred, alpha=0.3, s=10)
                lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
                plt.plot(lims, lims, 'r--')  # 对角线
                plt.xlim(lims)
                plt.ylim(lims)
                plt.xlabel("True Values")
                plt.ylabel("Predicted Values")
                title = 'Predicted vs True Expression' + name
                plt.title(title)
                plt.savefig(f"{title.replace(' ', '_')}.png")
                print("Saved scatter plot.")
            if return_var_ratio:        
                # ===== Cell Embedding =====
                all_cell_embeddings = torch.cat(all_cell_embeddings, dim=0).numpy()
                pca_Z = PCA(n_components=min(50, all_cell_embeddings.shape[1]))
                pca_Z.fit(all_cell_embeddings)
                var_ratio = pca_Z.explained_variance_ratio_[0]
            try:
                pearson_r = pearsonr(expr_preds_all.numpy(), expr_labels_all.numpy())[0]
            except:
                pearson_r = 0.0
            ss_res = torch.sum((expr_preds_all - expr_labels_all) ** 2)
            ss_tot = torch.sum((expr_labels_all - torch.mean(expr_labels_all)) ** 2)
            r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0
            pred_mean = torch.mean(expr_preds_all).item()
            true_mean = torch.mean(expr_labels_all).item()
            pred_std = torch.std(expr_preds_all).item()
            true_std = torch.std(expr_labels_all).item()
        
            return {
                "accuracy": accuracy,
                "expr_mae": mae,
                "expr_rmse": rmse,
                "expr_smape": smape,
                "expr_pearsonr": pearson_r,
                "expr_r2": r2,
                "cell_embedding_var_ratio": var_ratio if return_var_ratio else -1.0,
                "pred_mean": pred_mean,
                "true_mean": true_mean,
                "pred_std": pred_std,
                "true_std": true_std
            }

        for step, batch in tqdm(enumerate(eval_loader), total=len(eval_loader), desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(self.device)          # (B, L)
            expr_values = batch["expr_values"].to(self.device)      # (B, L)
            attn_mask = batch["attention_mask"].to(self.device)     # (B, L)

            # 随机生成 mask 或者根据需要直接传 mask
            mask = self._generate_mask(expr_values, masked_ratio=1.0)

            expr_preds, global_pred = self.model(input_ids=input_ids, expr_values=expr_values, mask=mask)
            
            # 退化检测
            if return_var_ratio:
                cell_embeddings = core.decoder(input_ids=input_ids, expr_values=expr_values, return_cls=True)
                all_cell_embeddings.append(cell_embeddings.detach().cpu())

            # ==== expr evaluation ====
            all_expr_preds.append(expr_preds.detach().cpu())
            all_expr_labels.append(expr_values[mask.bool()].detach().cpu())
            if global_pred is not None:
                selected_pred = torch.gather(global_pred, 1, input_ids)  # (B, L)
                all_global_preds.append(selected_pred[attn_mask.bool()].detach().cpu())
            
            
        results_pred = evaluate_metrics(all_expr_preds, all_expr_labels, all_cell_embeddings if return_var_ratio else None, return_var_ratio, show_scatter, name="/predicted")
        if len(all_global_preds) > 0:
            results_global = evaluate_metrics(all_global_preds, all_expr_labels, all_cell_embeddings=None, return_var_ratio=None, show_scatter=None, name="/global")
        else:
            results_global = None
        
        self.model.train()
        return results_pred, results_global

        # logger.info(f"[Eval] Acc: {accuracy:.4f} | Expr MAE: {mae:.4f} | RMSE: {rmse:.4f} | SMAPE: {smape:.4f} | PearsonR: {pearson_r:.4f} | R²: {r2:.4f} | mean_diff: {(pred_mean - true_mean):.4f} | std_diff: {(pred_std - true_std):.4f}")

    def save_model(self, 
                   epoch: int = None, 
                   base_dir: str = "./checkpoints", 
                   save_trainer: bool = False, 
                   print_info: bool = True,
                   step: int = None,
                   global_step: int = None):
        if self.save_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if self.expr_name is None:
                self.save_dir = os.path.join(base_dir, timestamp)
            else:
                self.save_dir = os.path.join(base_dir, self.expr_name, timestamp)
            os.makedirs(self.save_dir, exist_ok=True)

        save_dir = self.save_dir
        
        model = unwrap_model(self.model)
        model.save_model(os.path.join(save_dir, f"model_{global_step}.pt"))
        
        if save_trainer:
            training_state = {
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": epoch,
                "steps": step,
                # "log_sigma_name": self.log_sigma_name,
                # "log_sigma_expr": self.log_sigma_expr
            }
            torch.save(training_state, os.path.join(save_dir, f"training_state_{global_step}.pt"))

        print(f"Model and training state saved to {save_dir}") if print_info else None

    def load_training_state(self, checkpoint_path: str, strict: bool = True, print_info: bool = True):
        """
        从已保存的训练状态中恢复 optimizer 和 scheduler 的参数。

        Args:
            checkpoint_path (str): 保存的 training_state_*.pt 文件路径
            strict (bool): 是否严格加载（若 key 不匹配是否报错）
            print_info (bool): 是否打印加载信息
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Training state file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                if strict:
                    raise RuntimeError(f"Failed to load optimizer state: {e}")
                else:
                    print(f"Warning: failed to load optimizer state (ignored): {e}")
        
        if "scheduler_state_dict" in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                if strict:
                    raise RuntimeError(f"Failed to load scheduler state: {e}")
                else:
                    print(f"Warning: failed to load scheduler state (ignored): {e}")
                    
        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"]
            print(f"Resumed from epoch {self.start_epoch}") if print_info else None
            
        if "steps" in checkpoint:
            self.start_steps = checkpoint["steps"]
            print(f"Resumed from step {self.start_steps}") if print_info else None
        
        if print_info:
            print(f"Training states successfully restored from {checkpoint_path}")

    def fine_tune(self,
                new_data,
                old_vocab_size: int = 0,
                num_epochs: int = 5,
                lr: float = 1e-5,
                lora_rank: int = 8,
                lora_alpha: int = 16,
                lora_dropout: float = 0.05,
                save: bool = True,
                save_dir: str = "./checkpoints_finetune",
                save_epoch_interval: int = 1):
        """
        Fine-tune model when new proteins are added.
        - new_data: AnnData, ProteinSequenceDataset, or DataLoader
        - num_epochs: number of epochs for fine-tuning
        - lr: learning rate for fine-tuning
        - save: whether to save checkpoints
        """
        from peft import get_peft_model, LoraConfig, TaskType

        core = unwrap_model(self.model)
        
        # ===== Step 1: Frozen old parameters ====== #
        for name, param in core.named_parameters():
            param.requires_grad = False  # 默认冻结全部
        
        # ===== Step 2: Unlock new embedding weights ====== #
        embed_weight = core.embedding.protein_embed.embedding.weight
        embed_weight.requires_grad = True
        def grad_mask_hook(grad):
            grad[:old_vocab_size] = 0.0  # 冻结旧 token
            return grad

        embed_weight.register_hook(grad_mask_hook)
        
        # ====== Step 3: Unlock classification head ====== #
        for name, param in core.named_parameters():
            if "cls_global_head" in name or "cls_proj" in name:
                param.requires_grad = True
                
        # ====== Step 4: Apply LoRA to attention layers ====== #
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["out_proj", "linear1", "linear2"],
            task_type=TaskType.FEATURE_EXTRACTION,
            bias="none"
        )

        core.decoder = get_peft_model(core.decoder, lora_config)
        
        core.decoder.print_trainable_parameters()
        print(f"Overrall Trainable params: {sum(p.numel() for p in core.parameters() if p.requires_grad)} || All params: {sum(p.numel() for p in core.parameters())} || Trainable%: {100 * sum(p.numel() for p in core.parameters() if p.requires_grad) / sum(p.numel() for p in core.parameters()):.4f}%") 

        # ====== Step 5: DataLoader ====== #
        if isinstance(new_data, AnnData):
            dataset = ProteinSequenceDataset(new_data, self.tokenizer)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        elif isinstance(new_data, ProteinSequenceDataset):
            dataloader = DataLoader(new_data, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        elif isinstance(new_data, DataLoader):
            dataloader = new_data
        else:
            raise ValueError("Unsupported data type for new_data")

        # ====== Step 6: Optimizer and Scheduler ====== #
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, core.parameters()), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloader))

        # ====== Step 7: Fine-tuning loop ====== #
        self.model.to(self.device)
        self.model.train()
        loss_history = []
        for epoch in range(num_epochs):
            total_loss = 0.0
            for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Fine-tune Epoch {epoch+1}"):
                input_ids = batch["input_ids"].to(self.device)
                expr_values = batch["expr_values"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device)

                mask = self._generate_mask(expr_values, masked_ratio=self.masked_ratio)

                expr_preds, global_pred = self.model(input_ids=input_ids, expr_values=expr_values, mask=mask)
                mask_expr_values = expr_values[mask.bool()]
                loss_expr = (self.loss(expr_preds, mask_expr_values)).sum() / mask.sum()
                
                loss_global = 0.0
                if global_pred is not None:
                    selected_pred = torch.gather(global_pred, 1, input_ids)  # (B, L)
                    loss_global = (self.loss(selected_pred[attn_mask.bool()], expr_values[attn_mask.bool()])).sum() / attn_mask.sum()

                loss = loss_expr + loss_global
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(dataloader):.4f}")
            loss_history.append(total_loss / len(dataloader))
            # 早停策略：当loss的变化率小于1e-3时停止
            if epoch > 0:
                rel_change = abs(loss_history[-1] - loss_history[-2]) / (loss_history[-2] + 1e-8)
                if rel_change < 1e-3:
                    print(f"Loss change {rel_change:.6f} < 1e-3, stopping fine-tuning at epoch {epoch+1}")
                    break
            

            if save and (epoch + 1) % save_epoch_interval == 0:
                self.save_model(epoch + 1, save_dir)


if __name__ == "__main__":
    # 数据读取及排序
    file_dir = "/home/hubowen/data/final_data"
    train_data, test_data, merged_vars_names = load_dataset(file_dir, test_size = 0.2)

    print("Starting training...")
    # trainer.save_model()  # 保存初始模型
    model = ProteinLanguageModel.load_model_from_checkpoint("/home/hubowen/protein_llm/checkpoints/protein_language_model.pt")
    trainer = MaskTrainer(
        model=model,
        optimizer_cls="adamw",
        scheduler_cls="cosine",
        focal_loss=True,
        scheduler_kwargs={"T_max": 10}
    )

    trainer.train(train_data=train_data)


