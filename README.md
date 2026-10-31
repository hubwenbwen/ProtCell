# ProtCell

**A foundation model for single-cell proteomics.**

ProtCell learns shared cell states and protein relationships across heterogeneous protein panels using a protein codebook, continuous-abundance embeddings and masked language modeling. It connects fragmented profiles through cell annotation and integration, extends them through protein imputation and inference, and guides the selection of informative protein panels. The study brings together approximately 380 million cells from 98 datasets for pretraining and evaluation.

## Environment setup

ProtCell was pretrained on Moore Threads MTT S5000 GPUs using MUSA. Use Python 3.10 with the Moore Threads driver, MUSA runtime, and matching PyTorch and `torch_musa` packages installed.

```bash
git clone https://github.com/hubwenbwen/ProtCell.git
cd ProtCell
```

In your MUSA environment, install the project dependencies while keeping the MUSA-compatible PyTorch build (Bash):

```bash
python -m pip install -r <(sed '/^torch==/d' requirements.txt)
python -c "import torch, torch_musa; print(torch.musa.is_available())"
python -m ipykernel install --user --name protein_llm --display-name "ProtCell (Python 3.10)"
```

The device check should print `True`.

## Pretraining on MUSA (MTT S5000)

[train_musa.py](train_musa.py) runs masked protein-abundance pretraining with BF16 mixed precision and MCCL distributed training. Configure the model, optimizer, batch size, and training schedule in [configs/pretrain_musa.yaml](configs/pretrain_musa.yaml).

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

Training settings are defined in the [MUSA YAML configuration](configs/pretrain_musa.yaml). Checkpoints are saved under `checkpoints/protcell_pretrain/` and TensorBoard logs under `logs/protcell_pretrain/`.

For multi-GPU pretraining on a single S5000 node, use the same arguments with `torchrun` (set `--nproc_per_node` to the number of available GPUs):

```bash
torchrun --standalone --nproc_per_node=2 train_musa.py \
  --config_path configs/pretrain_musa.yaml \
  base.metadata_path=./databank_pretrain/metadata.json \
  base.tokenizer_path=./data_info/tokenizer_all_vars.json \
  base.expr_name=protcell_pretrain
```

## Pretraining on NVIDIA GPUs (CUDA)

Use a CUDA-enabled PyTorch environment with Python 3.10, then install the dependencies:

```bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

Prepare the DataBank shards as described above. From the repository root, run [train.py](train.py) with the [CUDA configuration](configs/large-scale-new-10_config.yaml).

Single GPU:

```bash
mkdir -p protcell_pretrain_cuda
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config_path configs/large-scale-new-10_config.yaml \
  base.metadata_path=./databank_pretrain/metadata.json \
  base.tokenizer_path=./data_info/tokenizer_all_vars.json \
  base.expr_name=protcell_pretrain_cuda \
  trainer.device=cuda \
  databank.cache_size=8
```

Multiple GPUs on one node (two GPUs shown):

```bash
mkdir -p protcell_pretrain_cuda
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  --config_path configs/large-scale-new-10_config.yaml \
  base.metadata_path=./databank_pretrain/metadata.json \
  base.tokenizer_path=./data_info/tokenizer_all_vars.json \
  base.expr_name=protcell_pretrain_cuda \
  trainer.device=cuda \
  databank.cache_size=8
```

Set `CUDA_VISIBLE_DEVICES` to the GPU IDs to use and `--nproc_per_node` to their count. Adjust `trainer.batch_size` and `trainer.gradient_accumulation_steps` for your GPU memory. The configuration uses BF16; use `trainer.precision=float32` on GPUs without BF16 support. Checkpoints are saved under `checkpoints/protcell_pretrain_cuda/` and TensorBoard logs under `logs/protcell_pretrain_cuda/`.

## Data and pretrained model

Download `protcell_example_data.tar.gz`, `protcell_pretrained.pt` and `protcell_pretrained.json` from the [v1.0.0 release](https://github.com/hubwenbwen/ProtCell/releases/tag/v1.0.0). Place the downloaded files in the repository root, then run:

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
