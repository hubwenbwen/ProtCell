from databank.DataBankReader import DataBankReader
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any


class DataBankDataset(Dataset):
    """
    PyTorch Dataset for reading from DataBank shards managed by DataBankReader.

    Each sample is a dict:
        {
            "expr": torch.FloatTensor(global_dim,),
            "raw_local": torch.FloatTensor(local_dim,),
            "dataset": str,
            "cell_idx": int,
            "shard_idx": int
        }

    Args:
        reader (DataBankReader): the metadata manager providing shard info & vocab.
        preload (bool): if True, preload all shard headers (not expression matrices).
        cache_size (int): number of shards to cache in memory (simple LRU).
    """

    def __init__(
        self,
        reader: DataBankReader,
        preload: bool = False,
        shard_filter: Optional[List[int]] = None,
        cache_size: int = 4,
    ):
        self.reader = reader
        self.global_dim = reader.vocab_size
        self.cache_size = cache_size

        # flatten all shards info
        if shard_filter is not None:
            self.shards = [reader.shards[i] for i in shard_filter]
        else:
            self.shards = reader.shards
        self.cum_counts = np.cumsum([s["num_cells"] for s in self.shards])
        self.total_cells = int(self.cum_counts[-1]) if len(self.cum_counts) > 0 else 0

        # simple cache for loaded shard data
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_order: List[str] = []  # maintain order for LRU eviction

        if preload:
            for s in self.shards:
                _ = self._load_shard(s["path"])

    def __len__(self):
        return self.total_cells

    # ---------- Internal helpers ---------- #

    def _locate(self, global_idx: int):
        """Find which shard & local row a global cell index belongs to."""
        shard_idx = int(np.searchsorted(self.cum_counts, global_idx, side="right"))
        shard = self.shards[shard_idx]
        start = 0 if shard_idx == 0 else int(self.cum_counts[shard_idx - 1])
        local_row = int(global_idx - start)
        return shard, shard_idx, local_row

    def _load_shard(self, full_path: str):
        """Load .npz shard and cache."""
        if full_path in self._cache:
            return self._cache[full_path]

        data = np.load(full_path, allow_pickle=True)
        expr = data["expr"]  # shape: (N, local_dim)
        pe_index = data["pe_index"]  # shape: (N, local_dim)
        local_proteins = list(data["local_proteins"]) if "local_proteins" in data else None

        # simple LRU eviction
        if len(self._cache) >= self.cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]
        self._cache[full_path] = {"expr": expr, "local_proteins": local_proteins, "pe_index": pe_index}
        self._cache_order.append(full_path)
        return self._cache[full_path]

    # ---------- PyTorch Dataset interface ---------- #

    def __getitem__(self, idx: int):
        if idx < 0:
            idx = self.total_cells + idx
        if idx < 0 or idx >= self.total_cells:
            raise IndexError(f"Index {idx} out of range (0, {self.total_cells-1})")

        shard, shard_idx, local_row = self._locate(idx)
        shard_data = self._load_shard(shard["path"])
        expr_values = shard_data["expr"][local_row]  # (local_dim,)
        input_ids = shard["protein_map"]  
        pe_index = shard_data["pe_index"][local_row]
        
        expr_values, input_ids = zip(*[
            (v, i)
            for v, i in zip(expr_values, input_ids)
            if v >= 0
        ])
        expr_values, input_ids = list(expr_values), list(input_ids)


        sample = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "expr_values": torch.tensor(expr_values, dtype=torch.float),
            "dataset": shard["dataset"],
            "idx": local_row,
            "shard_idx": shard_idx,
            "pe_index": torch.tensor(pe_index, dtype=torch.long),
        }
        return sample

    def shuffle_shards(self, epoch_seed: int):
        rng = np.random.default_rng(epoch_seed)
        self._shard_perm = rng.permutation(len(self.shards)).tolist()
        return self._shard_perm
    
    def assign_shards(self, rank: int, world_size: int):
        """将随机化后的 shards 平均分配给各 rank"""
        num_shards = len(self._shard_perm)
        shards_per_rank = num_shards // world_size
        extras = num_shards % world_size

        start = rank * shards_per_rank + min(rank, extras)
        end = start + shards_per_rank + (1 if rank < extras else 0)
        shard_subset = [self.shards[i] for i in self._shard_perm[start:end]]
        return shard_subset

    def get_rank_dataset(self, rank: int, world_size: int, epoch: int, 
                        cache_size: Optional[int] = None,
                        device: Optional[torch.device] = None,
                        distributed: bool = False):
        self.shuffle_shards(epoch)
        rank_shards = self.assign_shards(rank, world_size)
        rank_dataset = RankShardDataset(rank_shards, cache_size or self.cache_size)
        local_samples = len(rank_dataset)
        if distributed and device is not None:
            # 收集所有 rank 的样本数
            samples_tensor = torch.tensor([local_samples], dtype=torch.long, device=device)
            torch.distributed.all_reduce(samples_tensor, op=torch.distributed.ReduceOp.MIN)
            min_samples = samples_tensor.item()
        else:
            min_samples = local_samples
        
        return rank_dataset, min_samples

# ---------- 子层：rank 专属数据访问 ----------
class RankShardDataset(Dataset):
    def __init__(self, shards: List[Dict[str, Any]], cache_size: int = 4):
        self.shards = shards
        self.cache_size = cache_size

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_order: List[str] = []

        # 累积 cell 计数，用于索引定位
        self._counts = [s["num_cells"] for s in shards]
        self._cum_counts = np.cumsum(self._counts)
        self.total_cells = int(self._cum_counts[-1]) if len(self._cum_counts) > 0 else 0

    def __len__(self):
        return self.total_cells

    # ---------------- 内部工具函数 ---------------- #

    def _locate(self, idx: int):
        shard_idx = int(np.searchsorted(self._cum_counts, idx, side="right"))
        shard = self.shards[shard_idx]
        start = 0 if shard_idx == 0 else int(self._cum_counts[shard_idx - 1])
        local_row = int(idx - start)
        return shard, shard_idx, local_row

    def _load_shard(self, full_path: str):
        """带简单 LRU cache 的 shard 加载"""
        if full_path in self._cache:
            return self._cache[full_path]

        data = np.load(full_path, allow_pickle=True)
        expr = data["expr"]
        pe_index = data["pe_index"]
        local_proteins = list(data["local_proteins"]) if "local_proteins" in data else None

        # LRU 机制
        if len(self._cache) >= self.cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        self._cache[full_path] = {"expr": expr, "local_proteins": local_proteins, "pe_index": pe_index}
        self._cache_order.append(full_path)
        return self._cache[full_path]

    # ---------------- PyTorch Dataset 接口 ---------------- #

    def __getitem__(self, idx: int):
        if idx < 0:
            idx = self.total_cells + idx
        if idx < 0 or idx >= self.total_cells:
            raise IndexError(f"Index {idx} out of range (0, {self.total_cells-1})")

        shard, shard_idx, local_row = self._locate(idx)
        shard_data = self._load_shard(shard["path"])

        expr_values = shard_data["expr"][local_row]
        pe_index = shard_data["pe_index"][local_row]
        input_ids = shard["protein_map"]

        expr_values, input_ids = zip(*[
            (v, i)
            for v, i in zip(expr_values, input_ids)
            if v >= 0
        ])
        expr_values, input_ids = list(expr_values), list(input_ids)


        sample = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "expr_values": torch.tensor(expr_values, dtype=torch.float),
            "dataset": shard["dataset"],
            "idx": local_row,
            "shard_idx": shard_idx,
            "pe_index": torch.tensor(pe_index, dtype=torch.long),
        }
        return sample

if __name__ == "__main__":
    reader = DataBankReader("./databank_out")
    print(reader.summary())

    dataset = DataBankDataset(reader, preload=False)
    print(len(dataset))  # total cells

    # sample
    sample = dataset[466411]
    print(sample["expr_values"].shape)       # (global_dim,)
    
    dataset_rank0 = dataset.get_rank_dataset(rank=0, world_size=2, epoch=0)
    print(dataset_rank0[100])
    dataset_rank1 = dataset.get_rank_dataset(rank=1, world_size=2, epoch=0)
    print(dataset_rank1[100])
    dataset_rank0 = dataset.get_rank_dataset(rank=0, world_size=2, epoch=1)
    print(dataset_rank0[100])