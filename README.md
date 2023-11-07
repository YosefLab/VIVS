# VIVS

[![PyPI version](https://badge.fury.io/py/vivs.svg)](https://badge.fury.io/py/vivs)
[![Tests passing](https://github.com/YosefLab/VIVS/actions/workflows/main.yml/badge.svg)](https://github.com/YosefLab/VIVS/actions/workflows/main.yml)


VIVS (Variational Inference for Variable Selection) is a Python package to identify molecular dependencies in omics data.
Please refer to the [preprint](https://www.biorxiv.org/content/10.1101/2023.11.03.565520v1)
and to the project repository [here](https://github.com/YosefLab/VIVS) for more details.

## Installation

Currently, VIVS is only available on GitHub. To install the package, on Python >= 3.9, you can do the following:
1. install jax with GPU support (see [here](https://jax.readthedocs.io/en/latest/installation.html) for instructions)
2. install the package using pip:

```bash
pip install vivs
```
Alternatively, you can clone the repository and install the package in editable mode:

```bash
pip install -e .
```

## Basic usage

### Data format
To use VIVS in your project, import the data of intest in a scanpy AnnData object.
Right now, VIVS only supports assays where $X$ corresponds to gene expression counts.
In particular, make sure than the anndata contains raw counts and not normalized data.

The response(s) $Y$ of interest are expected to be stored in the `obsm` attribute of the anndata object, either as an array or as a dataframe.

VIVS may not scale to many thousands of genes.
In such a case, it is recommended to filter the genes before running VIVS, which can be done in the following way:

```python
from vivs import select_genes

adata = select_genes(adata, n_top_genes=2000)

```


### Model fitting and inference
VI-VS can be initialized and trained as follows:

```python
from vivs import VIVS

model = VIVS(
    adata,
    feature_obsm_key="my_obsm_key",
    xy_linear=False,
    xy_model_kwargs={"n_hidden": 8}
)
model.train_all()
```

Once the model is trained, feature significance can be computed as follows:

```python
res = model.get_hier_importance(n_clusters_list=[100, 200])
```
Here, `n_clusters_list` is a list of the number of clusters to consider for the hierarchical clustering of the features.

These results can be visualized using `plot_hier_importance`:

```python
model.plot_hier_importance(res, theme_kwargs=dict(figure_size=(15, 2))
```