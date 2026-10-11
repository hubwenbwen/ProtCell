# ProtCell

**A foundation model for single-cell proteomics.**

ProtCell learns shared cell states and protein relationships across heterogeneous protein panels using a protein codebook, continuous-abundance embeddings and masked language modeling. It connects fragmented profiles through cell annotation and integration, extends them through protein imputation and inference, and guides the selection of informative protein panels. The study brings together approximately 380 million cells from 98 datasets for pretraining and evaluation.

## Environment setup

Use Python 3.10. An NVIDIA GPU is recommended for pretraining and fine-tuning.

```bash
git clone https://github.com/hubwenbwen/ProtCell.git
cd ProtCell
conda create -n protcell python=3.10 pip -y
conda activate protcell
python -m pip install -r requirements.txt
python -m ipykernel install --user --name protein_llm --display-name "ProtCell (Python 3.10)"
```

For a specific CUDA or CPU build, install PyTorch 2.7.1 using the [official installation instructions](https://pytorch.org/get-started/previous-versions/#v271) before installing the requirements.

## Pretraining

Prepare preprocessed AnnData (`.h5ad`) files with harmonized protein names in `adata.var_names`. Set `adata.uns["technology"]` and `adata.uns["preprocess"]` to `CITE-seq` / `CLR` or `CyTOF` / `ARCSIN`, respectively. The training loader expects nonnegative abundances and treats negative values as invalid. Use the provided [protein vocabulary](data_info/tokenizer_all_vars.json) when building the training shards:

```bash
python -m databank.DataBankCreator \
  --input_dir /absolute/path/to/preprocessed_h5ad \
  --out_dir ./databank_pretrain \
  --tokenizer ./data_info/tokenizer_all_vars.json
```

Start pretraining from the repository root:

```bash
mkdir -p protcell_pretrain
python train.py --config_path configs/large-scale-new-10_config.yaml \
  base.metadata_path=./databank_pretrain/metadata.json \
  base.tokenizer_path=./data_info/tokenizer_all_vars.json \
  base.expr_name=protcell_pretrain \
  databank.cache_size=8
```

Training settings are defined in the [YAML configuration](configs/large-scale-new-10_config.yaml). Checkpoints are saved under `checkpoints/protcell_pretrain/` and TensorBoard logs under `logs/protcell_pretrain/`.

## Downstream examples

The notebooks demonstrate model loading, task-specific adaptation, prediction, evaluation and plotting.

| Notebook | Application |
|---|---|
| [Cluster](results/Cluster.ipynb) | Batch integration, clustering and UMAP visualization. |
| [Annotation](results/Annotation.ipynb) | Cell-type classification and annotation transfer. |
| [Imputation](results/Imputation.ipynb) | Masked protein-abundance reconstruction. |
| [Unobserved](results/Unobserved.ipynb) | Inference and analysis of unmeasured proteins. |
| [Panel Search](results/Panel_Search.ipynb) | Protein-panel selection and downstream evaluation. |

Set the data and checkpoint paths in each notebook, then run its cells with the **ProtCell (Python 3.10)** kernel. See the [notebook guide](results/README.md) for data requirements, run options and output locations.

```bash
jupyter lab
```
