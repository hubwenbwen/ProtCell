from genericpath import isdir

from sympy import true
from model import tokenizer
from preprocess.preprocess import load_dataset
from model.tokenizer import ProteinTokenizer
from model.model import ProteinLanguageModel
from trainer.ar_trainer import AutoRegressiveTrainer
from trainer.mask_trainer import MaskTrainer
from databank.DataBankReader import DataBankReader
from databank.DataBankDataset import DataBankDataset
from omegaconf import OmegaConf
import torch
from typing import Union, List
import os
import anndata as ad
from sklearn.model_selection import train_test_split

import logging

import torch.distributed as dist

import random
import numpy as np
import argparse

def setup_distributed():
    """初始化分布式环境（如果有的话）"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda", local_rank)
        distributed = True
        print(f"[Init] Distributed training on rank {dist.get_rank()} / {dist.get_world_size()} (GPU {local_rank})")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        distributed = False
        local_rank = 0
        print("[Init] Single GPU training.")
    return distributed, local_rank, device

def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        torch.manual_seed(seed + rank)

def reserve_gpu_memory(proportion=0.95):
    """
    动态占用当前显卡剩余显存的指定比例，防止其他框架（如 vLLM）抢占。
    """
    if not torch.cuda.is_available():
        return

    # 获取当前显卡的剩余显存 (free) 和 总显存 (total)
    device = torch.cuda.current_device()
    free_mem, total_mem = torch.cuda.mem_get_info(device)
    
    # 计算需要预占的大小（留一点点余量给系统驱动，比如取剩余空间的 95%）
    reserve_size = int(free_mem * proportion)
    
    print(f"检测到剩余显存: {free_mem / 1024**3:.2f} GB")
    print(f"正在预占: {reserve_size / 1024**3:.2f} GB 以保护训练进程...")
    
    # 瞬间分配并立即释放
    # 这一步会让 PyTorch 的 Caching Allocator 立即向系统申请这块内存并保持持有
    dummy = torch.empty(reserve_size, dtype=torch.uint8, device=f'cuda:{device}')
    del dummy
    
    # 强制清理缓存（可选，但通常 del 之后 PyTorch 依然会持有该池）
    # torch.cuda.empty_cache() # 注意：不要在这里调这个，调了就还给系统了
    
    print("显存预占成功，地盘已锁死。")

def train():
    
    # ===== Setup distributed training if available ===== #
    distributed = False
    local_rank = 0
    distributed, local_rank, device = setup_distributed()
    is_main_process = (not distributed) or (dist.get_rank() == 0)
    reserve_gpu_memory()

    # ===== Load configuration ===== #
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to the config file.")
    args, unknown = parser.parse_known_args()
    cfg = OmegaConf.load(args.config_path)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(unknown))  # for command line argso
    
    # ===== Set random seed ===== #
    set_random_seed(cfg.base.seed)
    
    if is_main_process:
        # ===== Save final config ===== #
        config_save_path = "./configs/"
        os.makedirs(config_save_path, exist_ok=True)
        save_path = os.path.join(config_save_path, cfg.base.expr_name + "_config.yaml")
        OmegaConf.save(cfg, save_path)
        print(f"Final config: {cfg}")
        
    # ===== Setup logging ===== #
    log_name = "trainer.log"
    if cfg.base.expr_name is not None:
        log_name = os.path.join(cfg.base.expr_name, log_name + f".rank{dist.get_rank()}" if distributed else log_name)
    
    # ===== Load DataBank ===== #
    reader = DataBankReader(cfg.base.metadata_path, technology_prerpocecss=cfg.databank.technology_preprocess)
    all_shards = list(range(reader.total_shards))
    if cfg.base.test_size > 0:
        train_shards, val_shards = train_test_split(all_shards, test_size=cfg.base.test_size, random_state=cfg.base.seed)
    else:
        train_shards = all_shards
        val_shards = None
    tokenizer = ProteinTokenizer.load_tokenizer(cfg.base.tokenizer_path)
    train_dataset = DataBankDataset(reader, shard_filter=train_shards, cache_size=cfg.databank.cache_size)
    val_dataset = DataBankDataset(reader, shard_filter=val_shards, cache_size=cfg.databank.cache_size) if val_shards is not None else None
    num_of_all_samples = train_dataset.total_cells
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(filename)s]:%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_name,
        filemode='w'
    )
    
    # ===== Load and model ===== #
    if cfg.base.model_path is None:
        model = ProteinLanguageModel(
            tokenizer=tokenizer,
            **cfg.model,
        )
    else:
        try:
            model = ProteinLanguageModel.load_model_from_checkpoint(cfg.base.model_path)
            print(f"Loaded model from {cfg.base.model_path}") if is_main_process else None
        except Exception as e:
            print(f"Failed to load model from {cfg.base.model_path}: {e}") if is_main_process else None
            return
    trainer = MaskTrainer(
        model=model,
        num_of_all_samples=num_of_all_samples,
        expr_name=cfg.base.expr_name,
        **cfg.trainer  
    )
    if cfg.base.training_state_path is not None:
        try:
            trainer.load_training_state(cfg.base.training_state_path, print_info=is_main_process)
            print(f"Loaded trainer state from {cfg.base.training_state_path}") if is_main_process else None
        except Exception as e:
            print(f"Failed to load trainer state from {cfg.base.training_state_path}: {e}") if is_main_process else None
            return

    if is_main_process:
        print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)}")
        print("Start training...")
    torch.autograd.set_detect_anomaly(True)
    
    # ===== Save initial model ===== #
    # model.save_model("initial_model.pt")
    # print("Initial model saved.")
    # model2 = ProteinLanguageModel.load_model_from_checkpoint("initial_model.pt")
    # print("Initial model loaded from checkpoint.")
    
    # ====== Start training ===== #
    try:
        trainer.train(train_data=train_dataset, 
                      test_data=val_dataset, 
                      distributed=distributed, 
                      local_rank=local_rank, 
                      seed=cfg.base.seed,
                      **cfg.training)
    except KeyboardInterrupt:
        print(f"Training interrupted.") if is_main_process else None
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    
    # ====== Final evaluation ===== #
    # results_pred, results_global = trainer.evaluate(val_dataset, return_var_ratio=True, show_scatter=True)
    # print("Without Focal Loss:", results["macro_f1"])
       
def fine_tune(
    model: ProteinLanguageModel = None,
    model_path: str = None,
    new_data: ad.AnnData = None,
    new_data_path: List = None,
    trainer: MaskTrainer = None,
    change_parameters: str = "new",
    num_epochs: int = 1,
    lr: float = 1.0e-5,
    save: bool = True,
):
    cfg = OmegaConf.load("config.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())  # for command line arg
    
    assert model is not None or model_path is not None, "Either model or model_path must be provided."
    assert new_data is not None or new_data_path is not None, "Either new_data or new_data_path must be provided."
    assert model is None or model_path is None, "Provide only model or model_path."
    assert new_data is None or new_data_path is None, "Provide only new_data or new_data_path."
    
    model = model if model is not None else None
    model = ProteinLanguageModel.load_model_from_checkpoint(model_path) if model_path is not None  and model is None else model

    new_data = new_data if new_data is not None else None
    new_data, _, new_vars_names = load_dataset(new_data_path, test_size=0.0) if new_data_path is not None and new_data is None else new_data
    
    added, old_vocab_size = model.add_new_proteins(new_vars_names)
    if added == 0:
        print("No new proteins to add. Exiting fine-tuning.")
        return
    
    if trainer is None:
        trainer = MaskTrainer(
            model=model,
            **cfg.trainer  
        )
    else:
        trainer = trainer
    
    if change_parameters == "all":
        old_vocab_size = 0  # fine-tune all parameters
    elif change_parameters == "new":
        pass  # only fine-tune new proteins
    print("Start fine tune...")
    trainer.fine_tune(
        new_data=new_data,
        old_vocab_size=old_vocab_size,
        num_epochs=num_epochs,
        lr=lr,
        save=save,
    )
        
if __name__ == '__main__':
    train()
    # fine_tune(
    #     model_path="/home/hubowen/work/protein_llm/checkpoints/ClsGlo+lr5e-4+weidec1e-3/2025-10-19_02-14-25/model_140.pt",
    #     new_data_path=["/home/hubowen/work/protein_data/cite_data/Dataset97.h5ad", "/home/hubowen/work/protein_data/cite_data/Dataset74.h5ad", "/home/hubowen/work/protein_data/cite_data/Dataset81.h5ad"],
    #     change_parameters="new",  # "all" or "new"
    # )
