import os
import json
import math
from pathlib import Path
from typing import List, Dict
import numpy as np
import anndata as ad
from tqdm import tqdm

from model.tokenizer import ProteinTokenizer


class DataBank:
    """Index and shard h5ad files for streaming training.

    Usage:
        db = DataBank(input_paths, out_dir)
        db.build_vocab(tokenizer_path="protein2id.json")  # optional
        db.index_files()
        db.shard_all(chunk_size=100000)
        db.save_metadata()
    """

    def __init__(self, h5ad_paths: List[str], out_dir: str):
        self.h5ad_paths = [str(p) for p in h5ad_paths]
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # populated later
        self.global_vocab: List[str] = []
        self.protein_to_id: Dict[str, int] = {}
        self.files_meta: Dict[str, Dict] = {}
        self.metadata_path = self.out_dir / "metadata.json"

    def build_vocab_from_tokenizer(self, tokenizer: ProteinTokenizer):
        """Load existing protein_to_id mapping (tokenizer-like)."""
        if tokenizer is None:
            return
        self.protein_to_id = tokenizer.protein_to_id
        # ensure ordering is stable: list index -> id keyed by value
        inv = {int(v): k for k, v in self.protein_to_id.items()}
        max_id = max(inv.keys()) if len(inv) > 0 else -1
        # build global_vocab as list where index == id
        vocab = [None] * (max_id + 1)
        for k, v in self.protein_to_id.items():
            vocab[int(v)] = k
        # drop None entries (if any)
        self.global_vocab = [p for p in vocab if p is not None]
        self.protein_to_id = {p: i for i, p in enumerate(self.global_vocab)}
        print(f"Loaded tokenizer vocab with {len(self.global_vocab)} proteins")

    def build_vocab_from_files(self, min_count: int = 1):
        """Scan all files and build a global protein vocab by occurrence count.

        min_count: only keep proteins seen in at least min_count files (useful to prune rare channels).
        """
        counter = {}
        for p in tqdm(self.h5ad_paths, desc="Scanning var_names"):
            adata = ad.read_h5ad(p, backed="r")
            local = [str(x) for x in adata.var_names]
            # count presence per file
            for prot in set(local):
                counter[prot] = counter.get(prot, 0) + 1
            try:
                adata.file.close()
            except Exception:
                pass

        # filter
        kept = [k for k, v in counter.items() if v >= min_count]
        kept = sorted(kept)
        self.global_vocab = kept
        self.protein_to_id = {p: i for i, p in enumerate(self.global_vocab)}
        print(f"Built global vocab with {len(self.global_vocab)} proteins (min_count={min_count})")

    def index_files(self):
        """Create per-file metadata: number of cells, local proteins, index_map -> positions in global_vocab."""
        if not self.global_vocab:
            raise RuntimeError("global_vocab is empty. Call build_vocab_* first.")

        for p in tqdm(self.h5ad_paths, desc="Indexing files"):
            adata = ad.read_h5ad(p, backed="r")
            local = [str(x) for x in adata.var_names]
            local_size = len(local)

            index_map = []
            for prot in local:
                if prot not in self.protein_to_id:
                    raise ValueError(f"Unknown protein '{prot}' found in file '{p}'. Please update protein_to_id mapping.")
                index_map.append(self.protein_to_id[prot])
            n_cells = adata.n_obs
            meta = {
                "path": str(p),
                "n_cells": int(n_cells),
                "n_local_proteins": int(local_size),
                "local_proteins": local,
                "index_map": index_map,
                "technology": adata.uns['technology'],
                "preprocess": adata.uns['preprocess'],
            }
            self.files_meta[str(p)] = meta
            try:
                adata.file.close()
            except Exception:
                pass

    def shard_all(self, chunk_size: int = 100000, overwrite: bool = False):
        """Shard each h5ad file into .npz chunks saved under out_dir/shards/<basename>/.

        Each shard file contains:
            - expr: (N_chunk, n_local_proteins) float32
            - obs_idx: list of cell ids (optional)
            - local_proteins: list of local protein names
            - pe_index: positional encoding indices
        And we update files_meta to include list of shard file paths with cell ranges.
        """
        shard_root = self.out_dir / "shards"
        shard_root.mkdir(parents=True, exist_ok=True)

        for p, meta in tqdm(self.files_meta.items(), desc="Sharding files"):
            in_path = meta["path"]
            basename = Path(in_path).stem
            out_dir = shard_root / basename
            out_dir.mkdir(parents=True, exist_ok=True)
            if "shards" in meta and not overwrite:
                print(f"Skipping {basename}, shards already exist. Use overwrite=True to re-shard.")
                continue

            adata = ad.read_h5ad(in_path, backed="r")
            n_cells = int(adata.n_obs)
            n_prots = int(adata.n_vars)
            local_prots = [str(x) for x in adata.var_names]

            shard_files = []
            n_shards = math.ceil(n_cells / chunk_size)
            for si in range(n_shards):
                start = si * chunk_size
                end = min((si + 1) * chunk_size, n_cells)
                # fetch dense slice; depending on backing mode, this reads portion from disk
                X = adata.X[start:end]
                # ensure numpy array (convert sparse if needed)
                if hasattr(X, "toarray"):
                    X = X.toarray()
                X = np.asarray(X, dtype=np.float32)
                pe_index_array = -np.ones((X.shape[0], n_prots), dtype=np.int32)
                for i, cell_expr in enumerate(X):
                    nonzero_mask = cell_expr != 0
                    expr_nonzero = cell_expr[nonzero_mask]
                    idx_nonzero = np.where(nonzero_mask)[0]
                    sorted_idx_in_nonzero = np.argsort(-expr_nonzero)
                    sorted_idx = idx_nonzero[sorted_idx_in_nonzero]
                    pe_index_array[i, :len(sorted_idx)] = sorted_idx
                obs_idx = list(range(start, end))

                shard_name = f"shard_{si:05d}.npz"
                shard_path = out_dir / shard_name
                np.savez_compressed(shard_path, expr=X, obs_idx=np.array(obs_idx, dtype=np.int32), local_proteins=np.array(local_prots, dtype=object), pe_index=pe_index_array)
                shard_files.append({"file": str(shard_path), "start": start, "end": end, "n_cells": X.shape[0]})

            meta["shards"] = shard_files
            try:
                adata.file.close()
            except Exception:
                pass

    def save_metadata(self):
        meta = {
            "global_vocab": self.global_vocab,
            "protein_to_id": self.protein_to_id,
            "files": self.files_meta,
        }
        with open(self.metadata_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved metadata to {self.metadata_path}")

    @classmethod
    def load_metadata(cls, metadata_path: str):
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        db = cls([], out_dir=str(Path(metadata_path).parent))
        db.global_vocab = meta["global_vocab"]
        db.protein_to_id = {k: int(v) for k, v in meta["protein_to_id"].items()}
        db.files_meta = meta["files"]
        db.metadata_path = Path(metadata_path)
        return db

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/home/hubowen/work/protein_data/test")
    parser.add_argument("--out_dir", type=str, default="./databank_out")
    parser.add_argument("--tokenizer", type=str, default="/home/hubowen/work/protein_llm/tokenizer_all_vars.json")
    parser.add_argument("--chunk_size", type=int, default=100000)
    args = parser.parse_args()

    if os.path.isdir(args.input_dir):
        h5ad_files = [str(p) for p in Path(args.input_dir).glob("*.h5ad")]
    elif os.path.isfile(args.input_dir):
        h5ad_files = [args.input_dir]
    else:
        raise ValueError(f"Invalid input_dir: {args.input_dir}")
    print(h5ad_files)
    db = DataBank(h5ad_files, args.out_dir)

    db.build_vocab_from_tokenizer(ProteinTokenizer.load_tokenizer(args.tokenizer))
    db.index_files()
    db.shard_all(chunk_size=args.chunk_size)
    db.save_metadata()
    print("Done.")
