# ProtCell downstream analysis notebooks

These notebooks accompany **ProtCell: a foundation model for single-cell proteomics** and demonstrate how to apply pretrained ProtCell representations to cell analysis, protein-profile completion and panel design. Each notebook includes data loading, model initialization, task-specific adaptation or training, prediction, quantitative evaluation and visualization.

## Notebook overview

| Notebook | Analysis | Manuscript figures |
|---|---|---|
| [Cluster.ipynb](Cluster.ipynb) | Batch integration, clustering, UMAP visualization and cluster-associated protein patterns. | Fig. 2a; Supplementary Fig. 2. |
| [Annotation.ipynb](Annotation.ipynb) | Cell-type classification within and across batches, annotation transfer and measured-marker analysis. | Fig. 2b,c; Supplementary Fig. 1. |
| [Imputation.ipynb](Imputation.ipynb) | Masked abundance reconstruction and protein-wise recovery. | Fig. 2d,e; Supplementary Fig. 3. |
| [Unobserved.ipynb](Unobserved.ipynb) | Inference of unmeasured cytotoxic proteins and analysis of cell states and NK subsets. | Fig. 2f–i. |
| [Panel_Search.ipynb](Panel_Search.ipynb) | Protein-panel selection, profile completion and acute myeloid leukaemia classification. | Fig. 2j,k. |

These applications connect cell states across fragmented panels, extend the information available in individual profiles and guide the selection of informative measurements.

## Getting started

### Environment

Follow the [repository installation instructions](../README.md#environment-setup) to create the Python 3.10 environment and install [requirements.txt](../requirements.txt). From the repository root, register the notebook kernel and start JupyterLab:

```bash
python -m ipykernel install --user --name protein_llm --display-name "ProtCell (Python 3.10)"
jupyter lab
```

Select the **ProtCell (Python 3.10)** kernel. An NVIDIA GPU is recommended for fine-tuning and protein inference. The Python environment includes the libraries used for statistical analysis and plotting. R-dependent kBET evaluation is optional; without those dependencies, the clustering notebook records kBET as unavailable (`NaN`).

### Paths and model checkpoints

The setup cell detects the project root and uses the bundled data and named pretrained checkpoint:

```python
from pathlib import Path

PROJECT_ROOT = Path("/absolute/path/to/ProtCell")
DATA_ROOT = PROJECT_ROOT / "results/data"
BASE_MODEL_PATH = PROJECT_ROOT / "checkpoints/protcell_pretrained.pt"
```

Place the pretrained checkpoint at `checkpoints/protcell_pretrained.pt`, or set `BASE_MODEL_PATH` to another checkpoint. Model checkpoints contain the architecture configuration and protein vocabulary. Set the task-specific checkpoint paths in the subsequent configuration cells when using a saved adapted model or classification head.

Run the notebook from a fresh kernel in cell order. The `RUN_*` switches enable training, inference, evaluation and plotting. Fine-tuning parameters, data splits and target proteins are specified in the relevant cells.

### Example data

Download the data bundle and checkpoint following the repository README, then extract the bundle from the project root so that inputs are placed under `results/data/`.

The examples use preprocessed AnnData (`.h5ad`) files. `adata.X` contains protein abundances, `adata.var_names` contains harmonized protein identities, and `adata.obs` contains the metadata required for each analysis. Use the CLR-preprocessed CITE-seq or arcsinh-preprocessed CyTOF inputs expected by the selected example. `ARCSIN` is the label used for the arcsinh data directory.

| Dataset | File relative to `DATA_ROOT` | Metadata used in these examples |
|---|---|---|
| Sacco et al. (2022) | `CLR_data/Dataset74.h5ad` | `Batch`, `coarsecelltype`. |
| Beneyto-Calabuig et al. (2023) | `CLR_data/Dataset107.h5ad` | `cohort`, `celltype`; panel evaluation additionally uses `patient`, `status`. |
| Kotliarov et al. (2020) | `CLR_data/Dataset83.h5ad` | `batch`, `celltype`; unmeasured-protein analysis additionally uses `sample`, `subtype`. |
| Peterson et al. (2021) | `ARCSIN_data/Dataset119.h5ad` | `condition` for the condition-based imputation split. |
| Wagner et al. (2019) | `ARCSIN_data/Dataset34.h5ad` | Cell identities for the half-cell imputation split. |

Panel Search uses the prepared Sparks et al. (2023) discovery, selection and test files under `DATA_ROOT/panel_search/`, together with discovery protein statistics and [panel/target definitions](data/panel_search/panel_definition.json). Keep these fixed splits and their statistics together. See the [data inventory](data/README.md) for all required files and [checksums](data/manifest.json).

## Analysis workflows

### Clustering and integration

[Cluster.ipynb](Cluster.ipynb) contains separate workflows for Sacco et al. (2022) and Beneyto-Calabuig et al. (2023). Each starts from the pretrained model, applies dataset-specific integration fine-tuning, extracts CLS embeddings, assigns clusters and evaluates cell organization and batch mixing. UMAPs are displayed by experimental batch, cluster and reference cell type.

The evaluation includes ARI, NMI, cluster silhouette, biological ASW, batch ASW, iLISI and cLISI, with kBET when available. Summary scores are calculated as:

```text
avgBio  = (ARI + NMI + Bio_ASW) / 3
avgBatch = (iLISI + Batch_ASW) / 2
overall = 0.6 × avgBio + 0.4 × avgBatch
```

The Sacco workflow additionally examines cluster composition and differential measured-protein patterns. Integration UMAPs use CLS embeddings; the cluster-marker visualization uses an expression-based neighbour graph with the cluster labels assigned from CLS embeddings.

To use saved adapted weights, set `RUN_SACCO_ADAPTATION=False` or `RUN_AML_ADAPTATION=False` and configure `SACCO_ADAPTED_MODEL_PATH` or `AML_ADAPTED_MODEL_PATH`, respectively. Keep embedding extraction enabled for subsequent analyses.

### Cell-type annotation

[Annotation.ipynb](Annotation.ipynb) freezes the pretrained ProtCell backbone and trains classification heads on Kotliarov et al. (2020). It includes five-fold within-batch evaluation, bidirectional cross-batch transfer and annotation transfer to Sacco et al. (2022). Accuracy, F1 scores and confusion matrices summarize classification performance; UMAPs, measured-marker plots and volcano plots characterize transferred cell states.

Use `RUN_FIVE_FOLD_TRAINING` and `RUN_CROSS_BATCH_TRAINING` to control classification-head training. When using saved cross-batch heads, set `RUN_CROSS_BATCH_TRAINING=False` and provide the corresponding files in `head_paths`.

### Measured-protein imputation

[Imputation.ipynb](Imputation.ipynb) adapts ProtCell on training cells and reconstructs masked abundances in held-out cells. Choose an entry in `EXPERIMENTS` using `EXPERIMENT`. Half-cell examples use consecutive halves of the dataset; batch- or condition-based examples define the split using the specified metadata column.

The notebook evaluates simultaneous masking of the configured quarter-protein targets and masking one protein at a time. Protein-wise outputs include Pearson and Spearman correlations, cosine similarity and abundance-error metrics, followed by recovery plots. The simultaneous targets are listed in `quarter_targets` and can be changed for a new analysis.

Set `RUN_FINETUNING=False` and configure `IMPUTATION_MODEL_PATH` to use an existing adapted checkpoint. `RUN_QUARTER_TEST` and `RUN_SINGLE_PROTEIN_TEST` control the two evaluation settings.

### Inference of unmeasured proteins

[Unobserved.ipynb](Unobserved.ipynb) uses Kotliarov et al. (2020) batch 1 for optional adaptation and batch 2 for inference. The queried proteins—GZMB, PRF1, KLRD1, TBX21 and EOMES—are absent from the target panel and present in the pretrained vocabulary. Adaptation uses `train_cls=False` to optimize the masked protein-position output.

The analysis produces:

- A cytotoxic-score violin plot across cell types (Fig. 2f).
- A heatmap of standardized inferred abundances, with equal weighting of biological samples (Fig. 2g).
- A volcano plot of paired sample-level differences in measured proteins, using paired t-test FDR (Fig. 2h).
- Split violin plots and paired sample-level contrasts for CD16-high and CD56-high/CD16-low NK subsets (Fig. 2i).

To infer from the adapted checkpoint used for the manuscript, set `RUN_FINETUNING=False`, select `IMPUTATION_MODEL_PATH=PAPER_CHECKPOINT_PATH` and keep `RUN_INFERENCE=True`. To perform a new adaptation, enable `RUN_FINETUNING`; the subsequent inference cell loads its saved checkpoint. The notebook records the checkpoint path and checksum alongside the predictions.

### Protein-panel search

[Panel_Search.ipynb](Panel_Search.ipynb) uses frozen ProtCell reconstruction to select 60 proteins from the Sparks et al. (2023) candidate panel. The search combines 20 cell resamples with 10 search seeds. Candidate panels are refined through marker exchanges, and the final panel is selected using reconstruction error on selection cells.

For a new search, set `RUN_PROTCELL_SEARCH=True` and `USE_NEW_PANEL=True`. To evaluate the panel used in the manuscript, set both switches to `False` and use the supplied panel and evaluation-target definitions.

Panel completion is evaluated using ridge, nearest-neighbour and extremely randomized tree decoders. The summaries report mean protein-wise Spearman correlation and median protein-wise RMSE in Sparks et al. (2023) and after transfer to Beneyto-Calabuig et al. (2023). A separate measured-protein classification analysis uses mapped 40-protein panels and summarizes AUROC and AUPRC by the median across held-out patients. `RUN_COMPLETION` and `RUN_AML` control these analyses.

## Outputs

By default, outputs are saved under this directory:

```text
outputs/
├── annotation/
├── cluster/
│   ├── Sacco2022/
│   └── Beneyto-Calabuig2023/
├── imputation/
├── unobserved/
└── panel_search/
```

Task directories contain figures, result tables, predictions or representations, and trained models where applicable. Metrics and plots are generated from the model predictions and analysis tables computed during execution.

To change the output root, set the environment variable before starting JupyterLab:

```bash
export PROTCELL_OUTPUT_DIR=/absolute/path/to/protcell_outputs
jupyter lab
```

Run the final cleanup cell to release the temporary plotting and JIT caches created by the notebook.
