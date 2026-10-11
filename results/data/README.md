# Notebook input data

Download `protcell_example_data.tar.gz` from the [v1.0.0 release](https://github.com/hubwenbwen/ProtCell/releases/tag/v1.0.0) and extract it from the project root, keeping the structure below. The notebook setup cells use this directory as `DATA_ROOT`.

| Study | Input file | Cells | Proteins | Notebooks |
|---|---|---:|---:|---|
| Sacco et al. (2022) | `CLR_data/Dataset74.h5ad` | 124,448 | 183 | Cluster, Annotation |
| Kotliarov et al. (2020) | `CLR_data/Dataset83.h5ad` | 53,201 | 79 | Annotation, Imputation, Unobserved |
| Beneyto-Calabuig et al. (2023) | `CLR_data/Dataset107.h5ad` | 100,620 | 140 | Cluster, Panel_Search |
| Peterson et al. (2021) | `ARCSIN_data/Dataset119.h5ad` | 420,000 | 35 | Imputation |
| Wagner et al. (2019) | `ARCSIN_data/Dataset34.h5ad` | 643,898 | 35 | Imputation |
| Sparks et al. (2023) | `panel_search/dataset101_discovery.h5ad` | 24,000 | 128 | Panel_Search |
| Sparks et al. (2023) | `panel_search/dataset101_selection.h5ad` | 7,200 | 128 | Panel_Search |
| Sparks et al. (2023) | `panel_search/dataset101_test.h5ad` | 24,000 | 128 | Panel_Search |

`CLR_data/` contains the CLR-preprocessed CITE-seq inputs; `ARCSIN_data/` contains the arcsinh-preprocessed CyTOF inputs. These files already contain the abundances and metadata used by the examples and should be loaded as provided.

For Panel Search, keep the Sparks et al. (2023) discovery, selection and test files together with:

- `panel_search/discovery_protein_statistics.npz`: discovery-derived protein means and scales, in the input protein order.
- `panel_search/panel_definition.json`: the 60-protein ProtCell panel, evaluation targets and mapped panel used for the completion examples.

The discovery, selection and test files retain the fixed donor splits used in the study. No resampling is needed to prepare the input files.

[manifest.json](manifest.json) records the filenames, studies, file sizes, matrix shapes and SHA-256 checksums. A pretrained checkpoint is supplied separately at `checkpoints/protcell_pretrained.pt`, relative to the project root.
