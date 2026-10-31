"""Masked protein-abundance pretraining on MUSA MTT S5000 GPUs."""
import torch_musa  # Register the MUSA device before importing the trainer and utilities.

from model.tokenizer import ProteinTokenizer
from model.model import ProteinLanguageModel
from trainer.mask_trainer_musa import MaskTrainer
from databank.DataBankReader import DataBankReader
from databank.DataBankDataset import DataBankDataset
from omegaconf import OmegaConf
import torch
import os
from sklearn.model_selection import train_test_split

import logging
import torch.distributed as dist
import argparse

import musa_utils as mu

def setup_distributed():
    """初始化分布式环境（自动检测 MUSA / CUDA）"""
    return mu.setup_distributed()

def set_random_seed(seed: int = 42):
    mu.set_random_seed(seed)

def reserve_gpu_memory(proportion=0.95):
    """预占当前 GPU 剩余显存（自动适配 MUSA/CUDA）"""
    mu.reserve_gpu_memory(proportion)

def train():

    # ===== Setup distributed training if available ===== #
    distributed = False
    local_rank = 0
    distributed, local_rank, device = setup_distributed()
    is_main_process = (not distributed) or (dist.get_rank() == 0)
    reserve_gpu_memory()

    # ===== Load configuration ===== #
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/pretrain_musa.yaml", help="Path to the config file.")
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

    # 确保日志目录存在
    log_dir = os.path.dirname(log_name)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(filename)s]:%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_name,
        filemode='w',
        force=True   # DDP 下每个 rank 都会重新配置各自的日志文件
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

if __name__ == "__main__":
    train()
