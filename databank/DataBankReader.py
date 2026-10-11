import os
import json
from typing import Dict, List, Any

class DataBankReader:
    """负责读取和管理Databank的元信息，不直接加载样本数据。"""

    def __init__(self, 
                 metadata_path: str,
                 technology_prerpocecss: Dict[str, str] = None):
        self.metadata_path = metadata_path
        self.technology_preprocess = technology_prerpocecss if technology_prerpocecss is not None else {}

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"metadata.json not found in {self.metadata_path}")

        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # 全局 vocab（protein_to_id）
        self.protein_to_id: Dict[str, int] = self.metadata.get("protein_to_id", {})
        self.id_to_protein: Dict[int, str] = {
            v: k for k, v in self.protein_to_id.items()
        }
        self.vocab_size = len(self.protein_to_id)

        # Shard 信息
        self.datasets = self.metadata.get("files", {})
        self.shards: List[Dict[str, Any]] = []

        for dataset_name, dataset_info in self.datasets.items():
            
            if self.technology_preprocess:
                data_technogoly = dataset_info["technology"]
                data_preprocess = dataset_info["preprocess"]
                if data_technogoly in self.technology_preprocess:
                    if self.technology_preprocess[data_technogoly] != data_preprocess:
                        continue  # skip this dataset
                else:
                    continue  # skip this dataset
            
            for shard_file in dataset_info['shards']:
                
                shard_path = shard_file['file']
                shard_idx = int(os.path.basename(shard_path).replace('shard_', '').replace('.npz', ''))

                self.shards.append({
                    'dataset': dataset_name,
                    'path': shard_path,
                    'index': shard_idx,
                    'protein_map': dataset_info['index_map'],
                    'technology': dataset_info['technology'],
                    'preprocess': dataset_info['preprocess'],
                    'num_cells': shard_file['n_cells']
                })

        self.total_shards = len(self.shards)

    def list_datasets(self) -> List[str]:
        return list(self.datasets.keys())

    def get_shard_info(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self.shards):
            raise IndexError(f"Shard index {idx} out of range.")
        return self.shards[idx]

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        return self.datasets.get(dataset_name, {})

    def summary(self) -> str:
        info = [
            f"Metadata Path: {self.metadata_path}",
            f"Total datasets: {len(self.datasets)}",
            f"Total shards: {self.total_shards}",
            f"Global vocab size: {self.vocab_size}",
        ]
        for name, ds in self.datasets.items():
            info.append(f" - {name}: {len(ds.get('shards', []))} shards, {ds.get('n_cells', '?')} cells")
        return "\n".join(info)

if __name__ == '__main__':
    reader = DataBankReader("./databank_out")
    print(reader.summary())
