# ProtCell

**A foundation model for single-cell proteomics.**

ProtCell learns shared cell states and protein relationships from heterogeneous protein panels. Each cell is represented as a variable-length cell sentence: a shared protein codebook supplies identity embeddings, continuous abundances are encoded by a multilayer perceptron, and a transformer learns their cellular context through masked abundance reconstruction and cell-level reconstruction.

The learned representations support three related applications:

- **Connect:** cell-type annotation, annotation transfer, clustering and batch integration.
- **Extend:** reconstruction of measured proteins and inference of unmeasured proteins represented in the pretrained vocabulary.
- **Guide:** selection of compact protein panels that retain information about omitted proteins and downstream cell states.

The study brings together approximately 380 million cells from 98 datasets for pretraining and downstream evaluation. This repository provides the model, pretraining code, task-specific fine-tuners and analysis notebooks.

## Contents

- [Environment setup](#environment-setup)
- [Data preparation](#data-preparation)
- [Pretraining](#pretraining)
- [Pretraining hyperparameters](#pretraining-hyperparameters)
- [Using a trained model](#using-a-trained-model)
- [Downstream examples](#downstream-examples)
- [Repository layout](#repository-layout)

## Environment setup

Use **Python 3.10** in a separate environment. Linux with an NVIDIA GPU is recommended for pretraining and fine-tuning. CPU execution is suitable for small inference examples.

```bash
git clone https://github.com/hubwenbwen/ProtCell.git
cd ProtCell

conda create -n protcell python=3.10 pip -y
conda activate protcell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

The requirements cover ProtCell pretraining, fine-tuning, statistical analysis, plotting and the five downstream notebooks. They include Scanpy and scIB for cell analysis, POT for optimal transport, PEFT for LoRA adaptation, and JupyterLab for notebook execution.

### Select a PyTorch build

The requirements specify PyTorch 2.7.1. To select its CUDA or CPU build explicitly, install PyTorch **before** installing the requirements. For example, for CUDA 12.8:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Replace `cu128` with `cu126`, `cu118` or `cpu` as appropriate for your machine. Check the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v271) for the corresponding builds and hardware requirements.

Check that the environment can import the model and access the GPU:

```bash
python -c "import torch, scanpy, scib, ot, peft; from model.model import ProteinLanguageModel; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### Notebook kernel

```bash
python -m ipykernel install --user --name protein_llm --display-name "ProtCell (Python 3.10)"
jupyter lab
```

Select this kernel when opening a notebook. R and kBET are not included in this setup. Without the R dependencies, the clustering notebook records kBET as unavailable (`NaN`); the other metrics are calculated separately.

## Data preparation

### Input format

Downstream examples use protein-expression matrices stored as AnnData `.h5ad` files:

- `adata.X`: cells by measured proteins, with the preprocessing expected by the model.
- `adata.var_names`: unique, harmonized protein names matching the model vocabulary.
- `adata.obs`: task-specific cell types, batches, samples, donors or conditions.

For pretraining, each file also needs two entries in `adata.uns`:

| Technology | `adata.uns["technology"]` | `adata.uns["preprocess"]` |
|---|---|---|
| CITE-seq | `CITE-seq` | `CLR` |
| CyTOF | `CyTOF` | `ARCSIN` |

`ARCSIN` is the metadata label used by the code for arcsinh preprocessing. The pretraining reader selects these exact technology/preprocessing pairs; it does **not** perform normalization. Prepare the transformed values before building the databank. The current pretraining loader retains nonnegative abundances and treats negative values as invalid, so use the nonnegative CLR-style representation expected by this workflow rather than a signed log-ratio matrix.

Keep each dataset's measured panel in its own file. Proteins absent from that panel do not need zero-filled columns. For inference of an unmeasured protein, the notebook adds its identity and masks its abundance explicitly.

### Protein vocabulary

The shared vocabulary is provided in [data_info/tokenizer_all_vars.json](data_info/tokenizer_all_vars.json). The token IDs used to create a databank must match those used by its model. All input protein names must be present in the selected vocabulary.

For pretraining a new model on your own protein set, create a vocabulary from your harmonized input files:

```python
from pathlib import Path
import anndata as ad
from model.tokenizer import ProteinTokenizer

input_files = sorted(Path("/absolute/path/to/preprocessed_h5ad").glob("*.h5ad"))
assert input_files, "No H5AD files found"
protein_names = set()
for path in input_files:
    data = ad.read_h5ad(path, backed="r")
    try:
        assert data.var_names.is_unique
        protein_names.update(data.var_names.astype(str))
    finally:
        data.file.close()

tokenizer = ProteinTokenizer(sorted(protein_names))
tokenizer.save_tokenizer("data_info/my_tokenizer.json")
```

When starting from an existing checkpoint, preserve its vocabulary and token order; use the task notebooks to adapt that model.

### Build training shards

Run commands from the repository root. The databank creator stores transformed expression matrices in compressed `.npz` shards and writes a `metadata.json` containing protein mappings, preprocessing labels, shard paths and cell counts.

```bash
python -m databank.DataBankCreator \
  --input_dir /absolute/path/to/preprocessed_h5ad \
  --out_dir ./databank_pretrain \
  --tokenizer ./data_info/tokenizer_all_vars.json \
  --chunk_size 100000
```

`--input_dir` accepts either one H5AD file or a directory of H5AD files. A directory is scanned at its top level. Input filenames should have distinct stems because those stems name the shard directories. Use `data_info/my_tokenizer.json` instead if you created a new vocabulary.

If CLR and ARCSIN data are stored in separate directories, build two databanks with the **same tokenizer**, then merge their metadata:

```bash
python -m databank.merge_metadata \
  --source ./databank_clr/metadata.json --select CITE-seq=CLR \
  --source ./databank_arcsin/metadata.json --select CyTOF=ARCSIN \
  --output ./data_info/my_pretraining_metadata.json
```

The merge checks vocabulary compatibility and shard cell counts. Relative shard paths are resolved from the working directory, so launch training from the repository root or use absolute shard paths.

Set the training manifest to the `metadata.json` produced by the databank creator or metadata merge. Provide the shard files referenced by its `files[*].shards[*].file` entries at accessible paths.

## Pretraining

The entry point is [train.py](train.py). It loads a YAML configuration and accepts additional OmegaConf overrides as `section.parameter=value` arguments.

### Single GPU

The following command uses the architecture configuration associated with the downstream examples and a small shard cache. Set the metadata path to the databank you prepared:

```bash
mkdir -p protcell_pretrain
python train.py --config_path configs/large-scale-new-10_config.yaml \
  base.metadata_path=./databank_pretrain/metadata.json \
  base.tokenizer_path=./data_info/tokenizer_all_vars.json \
  base.expr_name=protcell_pretrain \
  databank.cache_size=8
```

Creating the experiment directory first gives the training logger a destination for `trainer.log`. Replace the metadata argument with `./data_info/my_pretraining_metadata.json` for a merged databank. Use the matching tokenizer in each case.

For a shorter run, add overrides such as:

```bash
python train.py --config_path configs/large-scale-new-10_config.yaml \
  base.metadata_path=./databank_pretrain/metadata.json \
  base.tokenizer_path=./data_info/tokenizer_all_vars.json \
  base.expr_name=protcell_pretrain \
  databank.cache_size=8 \
  training.num_epochs=1 \
  trainer.batch_size=128 \
  trainer.gradient_accumulation_steps=1 \
  trainer.precision=float32
```

Use `bf16` on a GPU that supports bfloat16, `fp16` for CUDA mixed precision, or `float32` for full precision. The current training loop uses CUDA autocast; the commands above target GPU training.

### Multiple GPUs

```bash
mkdir -p protcell_pretrain
torchrun --standalone --nproc_per_node=4 train.py \
  --config_path configs/large-scale-new-10_config.yaml \
  base.metadata_path=./databank_pretrain/metadata.json \
  base.tokenizer_path=./data_info/tokenizer_all_vars.json \
  base.expr_name=protcell_pretrain \
  databank.cache_size=8
```

Set `--nproc_per_node` to the number of GPUs to use. The launcher initializes distributed training through NCCL. Approximate effective batch size is `trainer.batch_size × trainer.gradient_accumulation_steps × number of GPUs`.

### Checkpoints and logs

Training writes:

- The resolved configuration to `configs/<experiment>_config.yaml`.
- Text logs to `<experiment>/trainer.log` (with rank suffixes for distributed runs).
- TensorBoard events to `logs/<experiment>/<timestamp>/`.
- Model weights and optimizer/scheduler states to `checkpoints/<experiment>/<timestamp>/model_<step>.pt` and `training_state_<step>.pt`.

```bash
tensorboard --logdir ./logs
```

To continue from saved weights and training state, set both paths:

```bash
mkdir -p protcell_continue
python train.py --config_path configs/large-scale-new-10_config.yaml \
  base.metadata_path=./databank_pretrain/metadata.json \
  base.tokenizer_path=./data_info/tokenizer_all_vars.json \
  base.expr_name=protcell_continue \
  base.model_path=/absolute/path/to/model_9999.pt \
  base.training_state_path=/absolute/path/to/training_state_9999.pt \
  databank.cache_size=8
```

Use a matched pair from the same saved step. Setting only `base.model_path` initializes from saved weights with a new optimizer. A loaded checkpoint supplies its own model architecture and vocabulary.

## Pretraining hyperparameters

Values below refer to [configs/large-scale-new-10_config.yaml](configs/large-scale-new-10_config.yaml). The smaller [config.yaml](config.yaml) provides another starting configuration. Overrides are passed without a leading `--`, for example `trainer.lr=0.0005`.

### Data and initialization

| Parameter | Value in the example config | Meaning |
|---|---|---|
| `base.metadata_path` | `metadata.json` | Databank manifest; set to your actual path. |
| `base.tokenizer_path` | `tokenizer_all_vars.json` | Vocabulary JSON; the provided file is under `data_info/`. |
| `base.model_path` | `null` | Optional checkpoint for model initialization. |
| `base.training_state_path` | `null` | Optional optimizer, scheduler, epoch and step state. |
| `base.test_size` | `0` | Fraction of shards reserved for validation; `0` uses all selected shards. |
| `base.seed` | `42` | Random seed for initialization and shard splitting. |
| `base.expr_name` | `large-scale-new-10` | Name for configuration, logs and checkpoints. |
| `databank.cache_size` | `8192` | Number of shards cached per dataset instance; use a smaller value to limit RAM. |
| `databank.technology_preprocess` | `CyTOF: ARCSIN`, `CITE-seq: CLR` | Exact metadata pairs selected for training. |

Validation splitting is at the **shard** level, rather than by donor or dataset. The current epoch-level validation call is disabled in the training loop; changing `base.test_size` reserves shards but does not enable automatic validation reporting.

### Model

| Parameter | Value | Meaning |
|---|---|---|
| `model.d_model` | `256` | Transformer and protein-embedding width. |
| `model.hidden_dim` | `512` | Hidden width of the continuous-abundance embedding MLP. |
| `model.nhead` | `8` | Attention heads; must divide `d_model`. |
| `model.num_layers` | `8` | Transformer encoder layers. |
| `model.dim_feedforward` | `2048` | Transformer feed-forward width. |
| `model.dropout` | `0.15` | Transformer dropout. |
| `model.decoder_use_cls` | `true` | Include the cell-summary token. |
| `model.decoder_cls_cat_feat` | `false` | Concatenate the cell summary with protein representations for masked prediction. |
| `model.decoder_cls_global_predict` | `true` | Reconstruct the observed profile from the cell summary. |

The default embedding combination is summation. The example configuration enables both masked protein reconstruction and cell-level reconstruction; their losses are added during pretraining.

### Optimization

| Parameter | Value | Meaning |
|---|---|---|
| `trainer.device` | `cuda` | Device used by the trainer. |
| `trainer.batch_size` | `512` | Cells per batch on each GPU. |
| `trainer.gradient_accumulation_steps` | `32` | Mini-batches accumulated per optimizer update. |
| `trainer.lr` | `0.001` | Learning rate. |
| `trainer.weight_decay` | `0.0001` | Weight decay for Adam/AdamW. |
| `trainer.optimizer_cls` | `adam` | `adam`, `adamw` or `sgd`. |
| `trainer.scheduler_cls` | `cosine` | `cosine`, `step` or `linear`. |
| `trainer.scheduler_kwargs` | `{T_max: 10}` | Scheduler-specific settings; updated at optimizer steps. |
| `trainer.warmup_ratio` | `0.2` | Warmup duration calculated from the training-cell count and batch size. |
| `trainer.masked_ratio` | `0.15` | Fraction of eligible abundance positions masked during training. |
| `trainer.loss_type` | `mse` | `mse`, `smooth_l1` or `huber`. |
| `trainer.precision` | `bf16` | `bf16`, `fp16` or `float32`. |
| `trainer.log_steps` | `10` | Training-loss logging interval in mini-batches. |

Choose scheduler arguments for the selected scheduler: cosine uses `T_max`, step uses `step_size` and `gamma`, and linear uses `t_max`. In the current trainer, warmup steps are computed as `warmup_ratio × floor(training_cells / batch_size)` and the scheduler advances after optimizer updates.

### Run control

| Parameter | Value | Meaning |
|---|---|---|
| `training.num_epochs` | `500` | Total epoch limit. |
| `training.num_workers` | `2` | DataLoader worker processes. |
| `training.save` | `true` | Enable periodic step checkpoints. |
| `training.save_steps_interval` | `10000` | Checkpoint interval in mini-batch steps. |
| `training.log_dir` | `./logs` | TensorBoard log root. |

The YAML also contains `trainer.log_epoch`, `training.save_epoch_interval` and `training.save_dir`. Epoch-level reporting and checkpointing are currently disabled, and the saver currently uses `./checkpoints` as its root. Use the step interval and experiment name to control periodic checkpoint output. The trainer also attempts to save a checkpoint when it exits.

## Using a trained model

Set the checkpoint path to a trained ProtCell model on your machine. Model checkpoints include both the architecture configuration and the tokenizer. To obtain cell embeddings for a small batch:

```python
import anndata as ad
import numpy as np
import torch
from scipy import sparse
from model.model import ProteinLanguageModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ProteinLanguageModel.load_model_from_checkpoint(
    "/absolute/path/to/model.pt", device="cpu"
).to(device).eval()

adata = ad.read_h5ad("/absolute/path/to/preprocessed_proteins.h5ad")
assert adata.var_names.is_unique
unknown = set(adata.var_names) - set(model.tokenizer.get_all_proteins())
assert not unknown, f"Proteins absent from the model vocabulary: {unknown}"

batch = adata[:1024]
values = batch.X.toarray() if sparse.issparse(batch.X) else np.asarray(batch.X)
token_ids = torch.tensor(
    model.tokenizer.encode(batch.var_names.tolist()),
    dtype=torch.long, device=device,
).unsqueeze(0).expand(batch.n_obs, -1)
abundances = torch.as_tensor(values, dtype=torch.float32, device=device)

with torch.inference_mode():
    cell_embeddings = model.get_cell_embedding(
        input_ids=token_ids, expr_values=abundances, method="cls"
    ).cpu().numpy()

print(cell_embeddings.shape)  # (number of cells in this batch, d_model)
```

For larger datasets, iterate over batches as in the notebooks. Inference of proteins absent from a target panel uses protein identities already represented in the pretrained vocabulary; the [Unobserved notebook](results/Unobserved.ipynb) shows how to add and mask those targets.

## Downstream examples

The notebooks under [results](results) show data loading, model loading, task-specific training or fine-tuning, prediction, metrics and figures in explicit code cells.

| Notebook | Workflow | Example datasets |
|---|---|---|
| [Cluster](results/Cluster.ipynb) | Integration fine-tuning, CLS embeddings, UMAPs by batch/cluster/cell type, integration metrics and cluster-marker analysis. | Sacco2022 (`Dataset74`), Beneyto-Calabuig2023 (`Dataset107`). |
| [Annotation](results/Annotation.ipynb) | Freeze the pretrained backbone, train classification heads, evaluate within and across batches, then transfer annotations. | Kotliarov2020 (`Dataset83`), Sacco2022 (`Dataset74`). |
| [Imputation](results/Imputation.ipynb) | Fine-tune on training cells, reconstruct masked proteins, calculate correlations and errors, and plot recovery. | Peterson2021 (`Dataset119`), Wagner2019 (`Dataset34`), Kotliarov2020 (`Dataset83`). |
| [Unobserved](results/Unobserved.ipynb) | Infer unmeasured cytotoxic proteins, compute cell scores, and analyze measured-marker support and NK subsets. | Kotliarov2020 (`Dataset83`). |
| [Panel Search](results/Panel_Search.ipynb) | Search using frozen ProtCell, evaluate completion with independent decoders, and test measured-protein AML classification. | Sparks2023 (`Dataset101`), Beneyto-Calabuig2023 (`Dataset107`). |

Before running a notebook:

1. Set `PROJECT_ROOT` to your clone location and `DATA_ROOT` to your processed-data directory.
2. Set `BASE_MODEL_PATH` to the pretrained checkpoint you want to use. The examples currently point to `checkpoints/basemodel/large-scale-new-10/model_369999.pt`; provide that checkpoint or replace the path.
3. Check the data-loading cells for required `adata.obs` columns and task-specific inputs. Panel Search additionally uses prepared discovery/selection/test files under `DATA_ROOT/panel_search/` and the fixed panel/target definitions in `results/panel_definition.json`.
4. Choose the `RUN_*` switches, then run from a fresh kernel in order. When skipping training, explicitly provide the saved task checkpoint required by that notebook.

Outputs are written under `results/outputs/<task>/`. Set the `PROTCELL_OUTPUT_DIR` environment variable before running to choose another output root. The notebooks calculate predictions, metrics and plots during execution and export the resulting tables and figures.

The ridge, nearest-neighbour, extra-trees and logistic-regression models in Panel Search evaluate the information retained by the selected panel. Their implementation is supplied by scikit-learn.

## Repository layout

```text
ProtCell/
├── train.py                         # Pretraining entry point
├── config.yaml                      # Small starting configuration
├── configs/                         # Pretraining configurations
├── requirements.txt                 # Python dependencies
├── model/                           # Codebook, abundance embeddings and transformer
├── trainer/                         # Masked-abundance pretraining
├── databank/                        # Sharding, metadata and streaming datasets
├── finetuner/                       # Task-specific adaptation and prediction heads
├── preprocess/                      # Dataset-loading utilities
├── data_info/                       # Protein vocabulary
└── results/                         # Downstream notebooks, panel helpers and definitions
```
