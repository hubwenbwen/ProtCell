import anndata
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from typing import List, Union

def load_and_merge_datasets(file_paths, all_vars=None, batch_key_list: Union[List[str], None] = None):
    """
    读取多个AnnData数据集，合并，缺失蛋白补0
    :param file_paths: list of 文件路径（.h5ad）
    :return: 合并后的 AnnData
    """
    adatas = [anndata.read_h5ad(fp) for fp in file_paths]

    # 合并所有var_names，取并集，排序方便对齐
    if all_vars is None:
        all_vars = sorted(set().union(*[adata.var_names for adata in adatas]))
    else:
        with open(all_vars, "r") as f:
            all_vars = [line.strip() for line in f if line.strip()]

    merged_X = []
    merged_obs = []
    merged_vars = all_vars
    iterate = 0

    for (fp, adata) in zip(file_paths, adatas):
        dataset_name = os.path.splitext(os.path.basename(fp))[0]  # 用文件名作为 dataset 标记

        # 添加文件编号前缀，防止 obs_names 冲突
        adata.obs_names = [f"f{dataset_name}_{name}" for name in adata.obs_names]

        # === 处理 batch 信息 ===
        if 'batch' not in adata.obs.columns:
            if batch_key_list is not None:
                batch_key = batch_key_list[iterate]
                iterate += 1
                if batch_key not in adata.obs.columns:
                    raise ValueError(f"batch_key {batch_key} not in obs columns.")
                adata.obs["batch"] = adata.obs[batch_key].astype(str).apply(lambda x: f"{dataset_name}_{x}")
            else:
                # 如果没有 batch 列，就加一个 dataset 级别的 batch
                adata.obs["batch"] = dataset_name

        # 创建新 DataFrame 对齐变量
        df = pd.DataFrame(
            adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
            index=adata.obs_names,
            columns=adata.var_names
        )
        df_reindexed = df.reindex(columns=all_vars, fill_value=0)

        merged_X.append(df_reindexed)
        merged_obs.append(adata.obs)

    # 合并所有细胞
    merged_X_df = pd.concat(merged_X, axis=0)
    merged_obs_df = pd.concat(merged_obs, axis=0)

    # 转回 AnnData
    merged_adata = anndata.AnnData(
        X=merged_X_df.values,
        obs=merged_obs_df,
        var=pd.DataFrame(index=all_vars)
    )
    merged_adata.obs_names_make_unique()
    
    # compute_pe_index(merged_adata)

    return merged_adata, merged_vars

def split_train_test(adata, test_size=0.2, random_state=42):
    """
    按细胞划分训练测试集
    """
    train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=test_size, random_state=random_state)
    train_adata = adata[train_idx].copy()
    test_adata = adata[test_idx].copy()
    return train_adata, test_adata

def compute_pe_index(adata, key='pe_index'):
    """
    对每个细胞，跳过表达为0的蛋白，按表达量（含负数）由大到小排序，生成排序index（pe_index）
    :param adata: AnnData, X shape (cells, proteins)
    :return: list of np.array, 每个元素为该细胞的排序蛋白索引
    """
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    n_cells, n_proteins = X.shape
    
    # 初始化一个填充-1的数组
    pe_index_array = -np.ones((n_cells, n_proteins), dtype=int)
    
    for i, cell_expr in enumerate(X):
        # 找非0表达的蛋白
        nonzero_mask = cell_expr != 0
        # 对非0蛋白表达排序（降序）
        expr_nonzero = cell_expr[nonzero_mask]
        idx_nonzero = np.where(nonzero_mask)[0]
        # argsort降序，返回排序后非零蛋白在该细胞中的位置索引
        sorted_idx_in_nonzero = np.argsort(-expr_nonzero)
        # 映射回蛋白总索引
        sorted_idx = idx_nonzero[sorted_idx_in_nonzero]
        # 存储到数组中
        pe_index_array[i, :len(sorted_idx)] = sorted_idx
    
    adata.obsm[key] = pe_index_array
    return pe_index_array

def compute_protein_weight(adata, key: str = "protein_weight") -> np.ndarray:
    """
    根据蛋白在细胞中出现的频率的倒数计算权重。
    对于从未出现的蛋白，权重直接设为 0。
    """
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X  # 支持稀疏矩阵
    
    # 每个蛋白在多少个细胞中表达
    protein_nonzero_counts = (X != 0).sum(axis=0)  # shape: (n_proteins,)
    
    weights = np.zeros_like(protein_nonzero_counts, dtype=float)
    
    mask = protein_nonzero_counts > 0
    weights[mask] = protein_nonzero_counts.sum() / protein_nonzero_counts[mask]
    
    if mask.sum() > 0:
        weights[mask] /= weights[mask].sum()
    
    adata.var[key] = weights
    return weights

def compute_sample_weights(adata, key: str = "sample_weight") -> np.ndarray:
    """
    计算每个样本的权重，权重 = 该样本中蛋白质权重的最大值
    """
    protein_weights = adata.var['protein_weight'].values  # shape: (n_proteins,)
    sample_weights = []

    for i in range(adata.n_obs):
        # 当前样本的蛋白表达（bool 或数值）
        expr = adata.X[i].toarray().flatten() if hasattr(adata.X[i], 'toarray') else adata.X[i]
        # 找出非零蛋白的权重
        present_weights = protein_weights[expr > 0]
        if len(present_weights) > 0:
            sample_weights.append(np.max(present_weights))
        else:
            sample_weights.append(0.0)  # 该样本没有蛋白时默认 0

    sample_weights = np.array(sample_weights, dtype=float)
    adata.obs[key] = sample_weights
    return sample_weights

def load_dataset(file_dir, all_vars=None, test_size=0.2, random_state=42, pe_key='pe_index', weight_key="protein_weight", sample_key="sample_weight", print_info=True):
    if isinstance(file_dir, List):
        file_paths = file_dir
    elif os.path.isfile(file_dir):
        file_paths = [file_dir]
    else:
        file_paths = [os.path.join(file_dir, file) for file in os.listdir(file_dir) if file.endswith(".h5ad")]
    print(f"Starting to load and merge datasets from {file_dir}...") if print_info else None
    # TODO: 测试时只使用了前三个数据集，实际使用时需要修改
    merged_adata, merged_vars_names = load_and_merge_datasets(file_paths, all_vars=all_vars)
    print(f"Loaded merged AnnData with {merged_adata.n_obs} cells and {merged_adata.n_vars} proteins.") if print_info else None
    if test_size > 0:
        train_adata, test_adata = split_train_test(merged_adata, test_size=test_size, random_state=random_state)
        print(f"Splitted data into {len(train_adata)} cells for training and {len(test_adata)} cells for testing.") if print_info else None
    else:
        train_adata, test_adata = merged_adata, None
    compute_pe_index(train_adata, key=pe_key)
    compute_protein_weight(train_adata, key=weight_key)
    compute_sample_weights(train_adata, key=sample_key)
    if test_adata is not None:
        compute_pe_index(test_adata, key=pe_key)
        compute_protein_weight(test_adata, key=weight_key)
    print("Computed protein expression index and weight for training and testing data.") if print_info else None
    return train_adata, test_adata, merged_vars_names

if __name__ == "__main__" :
    # 示例调用：
    file_dir = "/home/hubowen/data/final_data"
    file_paths = [os.path.join(file_dir, file) for file in os.listdir(file_dir) if file.endswith(".h5ad")]
    merged_adata, merged_vars_names = load_and_merge_datasets(file_paths[:3])
    train_adata, test_adata = split_train_test(merged_adata)
    compute_pe_index(train_adata)
    compute_protein_weight(train_adata)
    print(train_adata.var_names)
    print(train_adata.var['protein_weight'])
