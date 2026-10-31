# ProtCell

**A foundation model for single-cell proteomics.**

ProtCell learns shared cell states and protein relationships across heterogeneous protein panels using a protein codebook, continuous-abundance embeddings and masked language modeling. It connects fragmented profiles through cell annotation and integration, extends them through protein imputation and inference, and guides the selection of informative protein panels. The study brings together approximately 380 million cells from 98 datasets for pretraining and evaluation.

## Environment setup

Use Python 3.10. ProtCell pretraining was performed on Moore Threads MTT S5000 GPUs using MUSA. Only pretraining was run on MUSA; downstream fine-tuning and evaluation were not validated on MUSA.

```bash
git clone https://github.com/hubwenbwen/ProtCell.git
cd ProtCell
conda create -n protcell python=3.10 pip -y
conda activate protcell
python -m pip install -r requirements.txt
python -m ipykernel install --user --name protein_llm --display-name "ProtCell (Python 3.10)"
```

For MTT S5000 pretraining, use a compatible Moore Threads driver, MUSA runtime, PyTorch and `torch_musa` installation. The `torch==2.7.1` pin in `requirements.txt` describes the existing CUDA/CPU environment and must not replace the vendor-compatible PyTorch build. In an activated MUSA environment, install the remaining dependencies with:

```bash
python -m pip install -r <(sed '/^torch==/d' requirements.txt)
python -c "import torch, torch_musa; print(torch.musa.is_available())"
```

The dependency command uses Bash process substitution. The availability check should print `True`. Exact versions of the original S5000 runtime are not recorded in the source patch. For CUDA/CPU downstream use, follow the existing requirements.

## Pretraining on MUSA (MTT S5000)

The MUSA pretraining code is extracted from `99_musa_patch`. Its core files are:

| File | Purpose |
|---|---|
| [train_musa.py](train_musa.py) | Pretraining entry point; loads DataBank shards, model and configuration. |
| [mask_trainer_musa.py](trainer/mask_trainer_musa.py) | Masked protein-abundance training, mixed precision, distributed training and checkpoints. |
| [musa_utils.py](musa_utils.py) | Device utilities, MCCL distributed backend and math SDPA workaround. |
| [pretrain_musa.yaml](configs/pretrain_musa.yaml) | Supplied MUSA configuration with BF16 enabled. |

The original `train.py` and CUDA trainer remain available. The separate MUSA entry point explicitly imports the MUSA trainer. Downstream patch files (`BaseFineTuner.py`, `BatchCorrection.py`, `Imputation.py`, `AddNewProtein.py`) and the path-specific evaluation script are excluded from this pretraining release. The copied trainer retains its original helper methods; their presence does not establish MUSA validation for downstream tasks.

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
python train_musa.py --config_path configs/pretrain_musa.yaml \
  base.metadata_path=./databank_pretrain/metadata.json \
  base.tokenizer_path=./data_info/tokenizer_all_vars.json \
  base.expr_name=protcell_pretrain \
  databank.cache_size=8
```

Training settings are defined in the [MUSA YAML configuration](configs/pretrain_musa.yaml). This is the configuration supplied with the patch, not a verified record of the full-scale training hyperparameters. Checkpoints are saved under `checkpoints/protcell_pretrain/` and TensorBoard logs under `logs/protcell_pretrain/`.

For multi-GPU pretraining on a single S5000 node, use the same arguments with `torchrun` (set `--nproc_per_node` to the number of available GPUs):

```bash
torchrun --standalone --nproc_per_node=2 train_musa.py \
  --config_path configs/pretrain_musa.yaml \
  base.metadata_path=./databank_pretrain/metadata.json \
  base.tokenizer_path=./data_info/tokenizer_all_vars.json \
  base.expr_name=protcell_pretrain
```

The utility selects MCCL on MUSA and disables flash/memory-efficient SDPA in favor of math attention. It also contains CUDA/CPU fallback paths; those paths do not imply that pretraining experiments were performed on those devices. These launch examples have not been rerun on S5000 as part of this repository update.

## Data and pretrained model

Download `protcell_example_data.tar.gz`, `protcell_pretrained.pt` and `protcell_pretrained.json` from the [v1.0.0 release](https://github.com/hubwenbwen/ProtCell/releases/tag/v1.0.0). Sign in to an account with repository access if the repository is private. Place the downloaded files in the repository root, then run:

```bash
tar -xzf protcell_example_data.tar.gz
mkdir -p checkpoints
mv protcell_pretrained.pt protcell_pretrained.json checkpoints/
```

The archive extracts to `results/data/`. Dataset details and checksums are listed in its [inventory](results/data/README.md) and [manifest](results/data/manifest.json).

## Downstream examples

The notebooks demonstrate model loading, task-specific adaptation, prediction, evaluation and plotting.

| Notebook | Application |
|---|---|
| [Cluster](results/Cluster.ipynb) | Batch integration, clustering and UMAP visualization. |
| [Annotation](results/Annotation.ipynb) | Cell-type classification and annotation transfer. |
| [Imputation](results/Imputation.ipynb) | Masked protein-abundance reconstruction. |
| [Unobserved](results/Unobserved.ipynb) | Inference and analysis of unmeasured proteins. |
| [Panel Search](results/Panel_Search.ipynb) | Protein-panel selection and downstream evaluation. |

Place the example data in `results/data/` and the pretrained checkpoint at `checkpoints/protcell_pretrained.pt`, then run the notebooks with the **ProtCell (Python 3.10)** kernel. See the [notebook guide](results/README.md) for data requirements, run options and output locations.

```bash
jupyter lab
```
