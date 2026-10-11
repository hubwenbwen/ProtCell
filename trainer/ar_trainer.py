# trainer.py

from sympy import false
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import numpy as np
from anndata import AnnData
from typing import List, Dict, Union, Callable
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from preprocess.preprocess import load_dataset
from scipy.stats import pearsonr
from model.model import ProteinLanguageModel

import os
import logging
from tqdm import tqdm
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='[%(filename)s]:%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='trainer.log',
    filemode='w'
)

logger = logging.getLogger(__name__)


class ProteinSequenceDataset(Dataset):
    def __init__(self, adata: AnnData, tokenizer, eos_expr: float = 0.0):
        self.adata = adata
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.eos_token = tokenizer.eos_token
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_token = tokenizer.pad_token
        self.eos_expr = eos_expr

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        # 获取该 cell 的蛋白名称、表达量、排序索引
        protein_names = list(self.adata.var_names)
        expr_values = self.adata.X[idx].toarray().flatten() if hasattr(self.adata.X[idx], 'toarray') else self.adata.X[idx]

        # 排序
        protein_names = [protein_names[i] for i in pe_index]
        expr_values = expr_values[pe_index]

        # 添加 eos
        protein_names.append(self.eos_token)
        expr_values = np.append(expr_values, self.eos_expr)
        pe_index = np.append(pe_index, 0)

        # 编码
        input_ids = [self.tokenizer.encode(name) for name in protein_names]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        expr_values = torch.tensor(expr_values, dtype=torch.float)
        pe_index = torch.tensor(pe_index, dtype=torch.long)

        return {
            "input_ids": input_ids,       # (L,)
            "expr_values": expr_values,   # (L,)
            "pe_index": pe_index,         # (L,)
            "idx": idx
        }

def collate_fn(batch):
    pad_token_id =  0 
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids, expr_values, pe_index, attn_mask, idx_list = [], [], [], [], []

    for x in batch:
        L = len(x["input_ids"])
        pad_len = max_len - L

        input_ids.append(F.pad(x["input_ids"], (0, pad_len), value=pad_token_id))
        expr_values.append(F.pad(x["expr_values"], (0, pad_len), value=0.0))
        pe_index.append(F.pad(x["pe_index"], (0, pad_len), value=-1))
        attn_mask.append(torch.cat([torch.ones(L), torch.zeros(pad_len)]))
        idx_list.append(x["idx"])

    return {
        "input_ids": torch.stack(input_ids),
        "expr_values": torch.stack(expr_values),
        "pe_index": torch.stack(pe_index),
        "attention_mask": torch.stack(attn_mask),
        "idx": idx_list
    }

class AutoRegressiveTrainer:
    def __init__(self, 
                model: nn.Module,
                device: str='cuda',
                batch_size: int=2048,
                lr: float=1e-5,
                weight_decay: float=0.01,
                frozen_epochs: int=0,
                focal_loss: bool=false,
                focal_loss_gamma: float=2.0,
                focal_loss_alpha: List[float]=None,
                log_epoch: int=1,
                log_steps: int=10,
                gradient_accumulation_steps: int=32,
                log_sigma_name: float=0.0,
                log_sigma_expr: float=0.0,
                optimizer_cls: Union[str, Callable] = 'adamw',
                scheduler_cls: Union[str, Callable] = 'cosine',
                scheduler_kwargs: dict = None,
                masked_ratio: float=0.15,
                ):

        self.model = model 
        self.tokenizer = model.tokenizer
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.focal_loss = focal_loss
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha
        self.log_epoch = log_epoch
        self.log_steps = log_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.log_sigma_name = torch.nn.Parameter(torch.tensor(log_sigma_name, device=device))
        self.log_sigma_expr = torch.nn.Parameter(torch.tensor(log_sigma_expr, device=device))

        # 保存优化器和调度器信息
        self.optimizer_cls = optimizer_cls
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.frozen_epochs = frozen_epochs

        # 初始化优化器和调度器
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.mse_loss = nn.MSELoss(reduction='none')
        
        self.writer = None
        
        self.save_dir = None
        
    def _get_optimizer(self):
        params = list(self.model.parameters())  + [self.log_sigma_name, self.log_sigma_expr]

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
                return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **self.scheduler_kwargs)
            elif self.scheduler_cls.lower() == 'step':
                return torch.optim.lr_scheduler.StepLR(self.optimizer, **self.scheduler_kwargs)
            elif self.scheduler_cls.lower() == 'linear':
                return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 - epoch / self.scheduler_kwargs.get("t_max", 10))
            else:
                raise ValueError(f"Unsupported scheduler: {self.scheduler_cls}")
        else:
            return self.scheduler_cls(self.optimizer, **self.scheduler_kwargs)

    def _focal_loss(self, input_logits, target_labels):
        """
        Args:
            input_logits: Tensor of shape (B, L, C)
            target_labels: Tensor of shape (B, L)
        Returns:
            focal loss: scalar
        """
        assert input_logits.dim() == 3 and target_labels.dim() == 2
        B, L, C = input_logits.shape

        # 1. Flatten
        input_logits = input_logits.reshape(-1, C)     # (B*L, C)
        target_labels = target_labels.reshape(-1)      # (B*L,)

        # 2. Mask pad tokens
        mask = (target_labels != self.tokenizer.pad_token_id)
        input_logits = input_logits[mask]
        target_labels = target_labels[mask]

        # 3. Get class weights
        target_weights = self.protein_weight[target_labels]   # (N,)

        # 4. Compute softmax and log-probs
        log_probs = F.log_softmax(input_logits, dim=-1)        # (N, C)
        probs = log_probs.exp()                                # (N, C)

        # 5. Gather target probs and log-probs
        target_probs = probs[range(len(target_labels)), target_labels]         # (N,)
        target_log_probs = log_probs[range(len(target_labels)), target_labels] # (N,)

        # 6. Modulating factor
        modulating_factor = (1.0 - target_probs) ** self.focal_loss_gamma      # (N,)

        # 7. Focal loss (weighted)
        focal_loss = -modulating_factor * target_log_probs * target_weights    # (N,)

        # 8. Final mean loss
        return focal_loss.mean()   

    def train(self, 
              train_data,
              test_data=None,
              num_epochs: int=10, 
              save: bool=false,
              save_dir: str='./checkpoints',
              save_epoch_interval: int=10,
              log_dir: str='./logs',
              ):
        
        log_dir = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.writer = SummaryWriter(log_dir=log_dir)
        
        self.model = nn.DataParallel(self.model).to(self.device)
        
        if isinstance(train_data, AnnData):
            if self.focal_loss and self.focal_loss_alpha is None:
                self.protein_weight = torch.tensor(
                    self.tokenizer.get_token_weights(train_data.var['protein_weight'].to_dict()),
                    dtype=torch.float32,
                    device=self.device
                )
            elif self.focal_loss:
                self.protein_weight = torch.tensor(self.focal_loss_alpha, device=self.device)
            dataset = ProteinSequenceDataset(train_data, self.tokenizer)
            # dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
            
            # 样本级长尾采样
            sample_weights = torch.tensor(train_data.obs["sample_weight"].values, dtype=torch.float)
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),  # 一个 epoch 采样次数
                replacement=True                  # 允许重复采样
            )
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                collate_fn=collate_fn
            )
            
        elif isinstance(train_data, ProteinSequenceDataset):
            dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        elif isinstance(train_data, DataLoader):
            dataloader = train_data
        else:
            raise ValueError("Unsupported data type for train_data")
        
        if test_data is not None:
            if isinstance(test_data, AnnData):
                test_dataset = ProteinSequenceDataset(test_data, self.tokenizer)
                test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
            elif isinstance(test_data, ProteinSequenceDataset):
                test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
            elif isinstance(test_data, DataLoader):
                test_dataloader = test_data
            else:
                raise ValueError("Unsupported data type for test_data")
        else:
            test_dataloader = None
        
        self.model.train()
        for epoch in range(num_epochs):
            for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}", leave=False):
                input_ids = batch["input_ids"].to(self.device)          # (B, L)
                expr_values = batch["expr_values"].to(self.device)      # (B, L)
                pe_index = batch["pe_index"].to(self.device)            # (B, L)
                idx = batch["idx"]
                attention_mask = batch["attention_mask"].to(self.device)

                name_logits, expr_preds = self.model(input_ids, expr_values, pe_index)

                # teacher forcing shift
                name_logits = name_logits[:, :-1, :]
                expr_preds = expr_preds[:, :-1]
                name_labels = input_ids[:, 1:]
                expr_labels = expr_values[:, 1:]
                mask = attention_mask[:, 1:]

                # compute loss
                if self.focal_loss:
                    name_loss = self._focal_loss(name_logits, name_labels)
                else:
                    name_loss = self.ce_loss(
                        name_logits.reshape(-1, name_logits.size(-1)),
                        name_labels.reshape(-1)
                    )
                    
                if epoch < self.frozen_epochs:
                    loss = name_loss
                else:
                    expr_loss = (self.mse_loss(expr_preds, expr_labels) * mask).sum() / mask.sum()
                    loss = (1 / (2 * torch.exp(self.log_sigma_name)**2)) * name_loss + \
                        (1 / (2 * torch.exp(self.log_sigma_expr)**2)) * expr_loss + \
                        self.log_sigma_name + self.log_sigma_expr
                    # loss = 0.8 * name_loss + 0.2 * expr_loss

                # backward
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                # logging for nan and inf gradients
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN or Inf"

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                
                if step % self.log_steps == 0:
                    if epoch >= self.frozen_epochs:
                        self.writer.add_scalar("train/log_sigma_name", self.log_sigma_name.item(), epoch * len(dataloader) + step)
                        self.writer.add_scalar("train/expr_loss", expr_loss.item(), epoch * len(dataloader) + step)
                        self.writer.add_scalar("train/log_sigma_expr", self.log_sigma_expr.item(), epoch * len(dataloader) + step)
                        self.writer.add_scalar("train/total_loss", loss.item(), epoch * len(dataloader) + step)
                    self.writer.add_scalar("train/name_loss", name_loss.item(), epoch * len(dataloader) + step)
                    self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], epoch * len(dataloader) + step)

            # epoch logging
            if (epoch + 1) % self.log_epoch == 0:
                if epoch < self.frozen_epochs:
                    logger.info(f"Epoch: {epoch + 1} | Name Loss: {name_loss.item():.4f}")
                else:
                    logger.info(f"Epoch: {epoch + 1} | Name Loss: {name_loss.item():.4f} | "
                                    f"Expr Loss: {expr_loss.item():.4f} | Total Loss: {loss.item():.4f}")
                    if test_dataloader is not None:
                        results = self.evaluate(test_dataloader)
                        # TensorBoard
                        self.writer.add_scalar("eval/name_accuracy", results['name_accuracy'], epoch)
                        self.writer.add_scalar("eval/macro_f1", results['macro_f1'], epoch)
                        self.writer.add_scalar("eval/expr_mae", results['expr_mae'], epoch)
                        self.writer.add_scalar("eval/expr_rmse", results['expr_rmse'], epoch)
                        self.writer.add_scalar("eval/expr_smape", results['expr_smape'], epoch)
                        self.writer.add_scalar("eval/expr_pearsonr", results['expr_pearsonr'], epoch)
                        self.writer.add_scalar("eval/expr_r2", results['expr_r2'], epoch)
                        self.writer.add_scalar("eval/expr_accuracy", results['expr_accuracy'], epoch)
        
            # save model
            if save and (epoch + 1) % save_epoch_interval == 0:
                self.save_model(epoch + 1, save_dir)
            
        self.writer.close()

    @torch.no_grad()
    def evaluate(self, 
                  eval_data, 
                  eval_batch_size: int = None
                  ) -> Dict[str, float]:
        self.model.eval()
        self.model.to(self.device)
        total_name_tokens = 0
        correct_name_tokens = 0

        all_expr_preds = []
        all_expr_labels = []
        
        all_name_preds = []
        all_name_labels = []

        if eval_batch_size is None:
            eval_batch_size = self.batch_size

        if isinstance(eval_data, AnnData):
            dataset = ProteinSequenceDataset(eval_data, self.tokenizer)
            eval_loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn)
        elif isinstance(eval_data, ProteinSequenceDataset):
            eval_loader = DataLoader(eval_data, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn)
        elif isinstance(eval_data, DataLoader):
            eval_loader = eval_data
        else:
            raise ValueError("Unsupported data type for eval_data")

        for step, batch in tqdm(enumerate(eval_loader), total=len(eval_loader), desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(self.device)          # (B, L)
            expr_values = batch["expr_values"].to(self.device)      # (B, L)
            pe_index = batch["pe_index"].to(self.device)            # (B, L)
            attention_mask = batch["attention_mask"].to(self.device)

            name_logits, expr_preds = self.model(input_ids, expr_values, pe_index)

            # shift
            name_logits = name_logits[:, :-1, :]        # (B, L-1, vocab_size)
            expr_preds = expr_preds[:, :-1]             # (B, L-1)
            name_labels = input_ids[:, 1:]              # (B, L-1)
            expr_labels = expr_values[:, 1:]            # (B, L-1)
            mask = attention_mask[:, 1:]                # (B, L-1)

            # ===== name accuracy =====
            name_preds = torch.argmax(name_logits, dim=-1)  # (B, L-1)
            matches = (name_preds == name_labels) & (name_labels != self.tokenizer.pad_token_id)
            correct_name_tokens += matches.sum().item()
            total_name_tokens += (name_labels != self.tokenizer.pad_token_id).sum().item()
            
            # ==== collect name preds/labels (for name macro F1) ====
            valid_mask = name_labels != self.tokenizer.pad_token_id
            all_name_preds.append(name_preds[valid_mask].detach().cpu())
            all_name_labels.append(name_labels[valid_mask].detach().cpu())

            # ==== expr evaluation ====
            all_expr_preds.append(expr_preds[mask.bool()].detach().cpu())
            all_expr_labels.append(expr_labels[mask.bool()].detach().cpu())

        name_accuracy = correct_name_tokens / total_name_tokens if total_name_tokens > 0 else 0.0

        # ====== expr evaluation ======
        expr_preds_all = torch.cat(all_expr_preds)
        expr_labels_all = torch.cat(all_expr_labels)

        mae = torch.mean(torch.abs(expr_preds_all - expr_labels_all)).item()
        rmse = torch.sqrt(torch.mean((expr_preds_all - expr_labels_all) ** 2)).item()
        smape = (2 * torch.abs(expr_preds_all - expr_labels_all) / (torch.abs(expr_preds_all) + torch.abs(expr_labels_all) + 1e-6)).mean().item()
        # ====== Accuracy ======
        epsilon = 0.1  # 你可以调整误差阈值，比如 0.1 或 0.2
        correct = torch.abs(expr_preds_all - expr_labels_all) < epsilon
        accuracy = correct.float().mean().item()
        try:
            pearson_r = pearsonr(expr_preds_all.numpy(), expr_labels_all.numpy())[0]
        except:
            pearson_r = 0.0
        ss_res = torch.sum((expr_preds_all - expr_labels_all) ** 2)
        ss_tot = torch.sum((expr_labels_all - torch.mean(expr_labels_all)) ** 2)
        r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0
        
        # ====== name macro F1 ======
        name_preds_all = torch.cat(all_name_preds).numpy()
        name_labels_all = torch.cat(all_name_labels).numpy()
        try:
            macro_f1 = f1_score(name_labels_all, name_preds_all, average='macro', zero_division=0)
        except:
            macro_f1 = 0.0

        # ====== Logging ======
        logger.info(f"[Eval] Protein Name Accuracy: {name_accuracy:.4f} | Macro F1: {macro_f1:.4f}")
        logger.info(f"[Eval] Acc: {accuracy:.4f} | Expr MAE: {mae:.4f} | RMSE: {rmse:.4f} | SMAPE: {smape:.4f} | PearsonR: {pearson_r:.4f} | R²: {r2:.4f}")


        self.model.train()
        return {
            "name_accuracy": name_accuracy,
            "macro_f1": macro_f1,
            "expr_mae": mae,
            "expr_rmse": rmse,
            "expr_smape": smape,
            "expr_pearsonr": pearson_r,
            "expr_r2": r2,
            "expr_accuracy": accuracy
        }

    def save_model(self, epoch: int = None, base_dir: str = "./checkpoints"):
        # 第一次保存时创建唯一的文件夹（只执行一次）
        if self.save_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.save_dir = os.path.join(base_dir, timestamp)
            os.makedirs(self.save_dir, exist_ok=True)

        save_dir = self.save_dir
        # 保存文件名
        # if epoch is not None:
        #     save_dir = os.path.join(self.save_dir, f"epoch_{epoch}")
        #     os.makedirs(epoch_dir, exist_ok=True)
        # else:
        #     save_dir = self.save_dir
        
        if isinstance(self.model, nn.DataParallel):
            self.model.module.save_model(os.path.join(save_dir, f"model_{epoch}.pt"))
        elif isinstance(self.model, ProteinLanguageModel):
            self.model.save_model(os.path.join(save_dir, f"model_{epoch}.pt"))
        else:
            raise ValueError("Unsupported model type for saving")

        # # 保存训练状态
        # training_state = {
        #     "log_sigma_name": self.log_sigma_name.detach().cpu(),
        #     "log_sigma_expr": self.log_sigma_expr.detach().cpu(),
        #     "optimizer_state_dict": self.optimizer.state_dict(),
        #     "scheduler_state_dict": self.scheduler.state_dict(),
        #     "epoch": epoch
        # }

        # torch.save(training_state, os.path.join(save_dir, "training_state.pt"))
        print(f"Model and training state saved to {save_dir}")

    def load_model_and_state(self, checkpoint_dir: str,
                            model_file: str = "model.pt",
                            trainer_file: str = "training_state.pt",
                            device: str = "cuda" if torch.cuda.is_available() else "cpu"):

        self.model = ProteinLanguageModel.load_model(os.path.join(checkpoint_dir, model_file)).to(device)
        self.tokenizer = self.model.tokenizer

        # # 加载训练状态
        # training_state_path = os.path.join(checkpoint_dir, trainer_file)
        # training_state = torch.load(training_state_path, map_location=device)

        # # 初始化 optimizer 和 scheduler（结构要与原训练时一致）
        # optimizer = torch.optim.AdamW(model.parameters())  # 加上参数如 lr=...
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 示例

        # optimizer.load_state_dict(training_state["optimizer_state_dict"])
        # scheduler.load_state_dict(training_state["scheduler_state_dict"])

        # log_sigma_name = training_state["log_sigma_name"].to(device)
        # log_sigma_expr = training_state["log_sigma_expr"].to(device)

        # print(f"Loaded model and state from {checkpoint_dir}")
        return self.model # optimizer, scheduler, log_sigma_name, log_sigma_expr

    @torch.no_grad()
    def name_evaluation(self, eval_data, top_n_plot: int = 30):
        self.model.eval()
        self.model.to(self.device)
        total_name_tokens = 0
        correct_name_tokens = 0
        
        all_name_preds = []
        all_name_labels = []

        eval_batch_size = self.batch_size

        if isinstance(eval_data, AnnData):
            dataset = ProteinSequenceDataset(eval_data, self.tokenizer)
            eval_loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn)
        elif isinstance(eval_data, ProteinSequenceDataset):
            eval_loader = DataLoader(eval_data, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn)
        elif isinstance(eval_data, DataLoader):
            eval_loader = eval_data
        else:
            raise ValueError("Unsupported data type for eval_data")

        for step, batch in tqdm(enumerate(eval_loader), total=len(eval_loader), desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(self.device)          # (B, L)
            expr_values = batch["expr_values"].to(self.device)      # (B, L)
            pe_index = batch["pe_index"].to(self.device)            # (B, L)
            attention_mask = batch["attention_mask"].to(self.device)

            name_logits, expr_preds = self.model(input_ids, expr_values, pe_index)

            # shift
            name_logits = name_logits[:, :-1, :]        # (B, L-1, vocab_size)
            expr_preds = expr_preds[:, :-1]             # (B, L-1)
            name_labels = input_ids[:, 1:]              # (B, L-1)
            expr_labels = expr_values[:, 1:]            # (B, L-1)
            mask = attention_mask[:, 1:]                # (B, L-1)

            # ===== name accuracy =====
            name_preds = torch.argmax(name_logits, dim=-1)  # (B, L-1)
            matches = (name_preds == name_labels) & (name_labels != self.tokenizer.pad_token_id)
            correct_name_tokens += matches.sum().item()
            total_name_tokens += (name_labels != self.tokenizer.pad_token_id).sum().item()
            
            # ==== collect name preds/labels (for name macro F1) ====
            valid_mask = name_labels != self.tokenizer.pad_token_id
            all_name_preds.append(name_preds[valid_mask].detach().cpu().numpy())
            all_name_labels.append(name_labels[valid_mask].detach().cpu().numpy())
        
        all_name_preds = np.concatenate(all_name_preds)
        all_name_labels = np.concatenate(all_name_labels)

        # 结果保存目录
        output_dir = os.path.join(self.save_dir or "./", "name_eval")
        os.makedirs(output_dir, exist_ok=True)

        # ====== 混淆矩阵 ======
        cm = confusion_matrix(
            all_name_labels, 
            all_name_preds, 
            labels=range(self.tokenizer.vocab_size)
        )
        np.save(os.path.join(output_dir, "confusion_matrix.npy"), cm)

        # ====== 分类报告 ======
        target_names = [self.tokenizer.decode(i) for i in range(self.tokenizer.vocab_size)]
        report_dict = classification_report(
            all_name_labels, all_name_preds,
            labels=range(self.tokenizer.vocab_size),
            target_names=target_names,
            zero_division=0,
            output_dict=True
        )
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(os.path.join(output_dir, "classification_report.csv"), encoding="utf-8")

        # ====== 绘制 top-N 混淆矩阵 ======
        supports = report_df["support"].iloc[:-3]  # 去掉 avg/total 行
        top_classes_idx = supports.sort_values(ascending=False).index[:top_n_plot]
        try:
            top_class_ids = [self.tokenizer.encode(cls) for cls in top_classes_idx]
        except:
            # 如果名字不是直接可转ID
            top_class_ids = [i for i in range(top_n_plot)]

        cm_top = cm[np.ix_(top_class_ids, top_class_ids)]

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_top, annot=False, fmt="d", cmap="Blues",
                    xticklabels=top_classes_idx, yticklabels=top_classes_idx)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Top-{top_n_plot} Classes Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_top{top_n_plot}.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    # 数据读取及排序
    file_dir = "/home/hubowen/data/final_data"
    train_data, test_data, merged_vars_names = load_dataset(file_dir, test_size = 0.2)

    # d_model = 128
    # tokenizer = ProteinTokenizer(merged_vars_names)
    # embedder = CombinedProteinEmbedding(num_proteins = tokenizer.vocab_size, d_model = d_model)
    # model = ProteinDecoderOnlyModel(vocab_size = tokenizer.vocab_size,
    #                                  embedding = embedder, 
    #                                  d_model = d_model, 
    #                                  nhead = 4, 
    #                                  num_layers=2, 
    #                                  dim_feedforward=256
    #                                  )


    print("Starting training...")
    # trainer.save_model()  # 保存初始模型
    model = ProteinLanguageModel.load_model_from_checkpoint("/home/hubowen/protein_llm/checkpoints/protein_language_model.pt")
    trainer = AutoRegressiveTrainer(
        model=model,
        optimizer_cls="adamw",
        scheduler_cls="cosine",
        focal_loss=True,
        scheduler_kwargs={"T_max": 10}
    )
    # trainer.load_model("/home/hubowen/protein_llm/checkpoints/2025-07-27_13-34-58/model_epoch_80.pt")
    # protein_language_model = ProteinLanguageModel(
    #     tokenizer=trainer.tokenizer,
    #     embedding=trainer.model.embedding,
    #     decoder=trainer.model,
    # )
    # protein_language_model.to(trainer.device)
    # protein_language_model.save_model("/home/hubowen/protein_llm/checkpoints/protein_language_model.pt")
    trainer.train(train_data=train_data)
    # print("Starting evaluation...")
    # results = trainer.evaluate(eval_data=test_data)
    # print("Evaluation results:", results)


