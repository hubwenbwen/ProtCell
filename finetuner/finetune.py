from finetuner.BatchCorrection import *
from preprocess.preprocess import load_and_merge_datasets
from model.model import ProteinLanguageModel
import anndata as ad
import numpy as np

# 定义映射关系
celltype_mapping = {
    # T Cells
    'CD4_Naive': 'CD4 T', 'CD4_EM': 'CD4 T', 'CD4_CM': 'CD4 T', 'CD4_Treg': 'CD4 T', 
    'CD4_Tfh': 'CD4 T', 'CD4_platelet_bind': 'CD4 T', 'CD4': 'CD4 T',
    'CD8_TEMRA': 'CD8 T', 'CD8_CM': 'CD8 T', 'CD8_EM': 'CD8 T', 'CD8_Naive': 'CD8 T', 
    'CD8_TRM': 'CD8 T', 'CD8_proliferating': 'CD8 T', 'CD8': 'CD8 T',
    'MAIT': 'MAIT', 'gdT-Vd2': 'gdT', 'T_Vd2': 'gdT', 'DNT': 'DNT', 'NKT': 'NKT',
    
    # B Cells
    'B_Naive': 'B Cell', 'B_Naive_Intermediate': 'B Cell', 'B_Mem': 'B Cell', 
    'B': 'B Cell', 'Plasmablast': 'Plasmablast',
    
    # Myeloid
    'Mono_Classical': 'Monocyte', 'Mono_NonClassical': 'Monocyte', 'Mono_Intermediate': 'Monocyte',
    'Mono': 'Monocyte', 'Mac-or-Mono': 'Monocyte', 'cDC': 'DC', 'pDC': 'DC', 'Neut': 'Neutrophil',
    
    # NK & Others
    'NK_CD56hiCD16lo': 'NK', 'NK_CD16hi': 'NK', 'NK_proliferating': 'NK', 'NK': 'NK',
    'Platelet': 'Platelet', 'HSPC': 'Progenitor', 'HSC': 'Progenitor', 'cKit+CD3-activated': 'Progenitor',
    'RBC': 'Erythrocyte', 'ILC': 'ILC'
}

# 需要剔除或标记为不明的类别
trash_types = ['dim', 'Mono-T-dblt', 'nan', np.nan]

def finalize_annotation(adata):
    # 1. 创建新列，默认拷贝细粒度标注
    adata.obs['unified_celltype'] = adata.obs['celltype'].astype(str)
    
    # 2. 如果 celltype 是 'nan'，尝试从 coarsecelltype 补全
    mask_missing = adata.obs['unified_celltype'].isin(['nan', 'None'])
    adata.obs.loc[mask_missing, 'unified_celltype'] = adata.obs.loc[mask_missing, 'coarsecelltype'].astype(str)
    
    # 3. 应用映射字典
    adata.obs['unified_celltype'] = adata.obs['unified_celltype'].map(
        lambda x: celltype_mapping.get(x, 'Unknown')
    )
    
    # 4. 处理特殊项：将原标注中的 dim 或 dblt 设为 Unknown 或直接过滤
    # 这里我们统一设为 'Unknown'
    adata.obs.loc[adata.obs['celltype'].isin(['Mono-T-dblt', 'dim']), 'unified_celltype'] = 'Unknown'
    
    return adata


if __name__ == '__main__':
    
    all_files = ["/data/hubowen/protein_data/CLR_data/Dataset107.h5ad", "/data/hubowen/protein_data/CLR_data/Dataset74.h5ad", "/data/hubowen/protein_data/CLR_data/Dataset101.h5ad", "/data/hubowen/protein_data/CLR_data/Dataset83.h5ad"]

    # adata_107, _ = load_and_merge_datasets(all_files[:1])
    
    adata_74, _ = load_and_merge_datasets(all_files[1:2])
    
    adata_101, _ = load_and_merge_datasets(all_files[2:3])

    # adata_83, _ = load_and_merge_datasets(all_files[3:4])
    
    # adata_74_batch1 = adata_74[adata_74.obs['Batch'] == 'B4chip3'].copy()
    # adata_101_batch1 = adata_101[adata_101.obs['Batch'] == 'Batch1'].copy()


    # adata_combined = ad.concat([adata_74_batch1, adata_101_batch1], join='outer')
    adata_combined = ad.concat([adata_74, adata_101], join='outer')
    adata_combined.X[np.isnan(adata_combined.X)] = 0.0
    adata_combined = finalize_annotation(adata_combined)
    
    model = ProteinLanguageModel.load_model_from_checkpoint("/home/hubowen/work/protein_llm/checkpoints/basemodel/large-scale-new-9/model_749999.pt")
    # model = ProteinLanguageModel.load_model_from_checkpoint("./checkpoints/BatCorr/BatCorr-MultiDataset-kmeans-3/finetune_5.pt")
    
    try:
        batchcorrection_finetuner.fit(
            num_epochs=100,
            save=True,
            save_dir="./checkpoints/BatCorr",
            save_epoch_interval=1,
            early_stop=False,
            lora_rank=4,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=["linear1", "linear2", "cls_proj.0", "cls_proj.3"],
            mnn_k=10,
            neg_ratio=0.25,
            num_clusters=20,
            quantile=0.8,
            tau_intra=0.1,
            lambda_intra=1.0e3
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    batchcorrection_finetuner.plot_loss_history(save_path="./figures/BatCorr_loss.png")
    