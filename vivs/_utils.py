import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid


def select_genes(
    adata,
    n_top_genes: int,
    preselected_genes: int = None,
    seed: int = 0,
):
    """Selects genes for analysis that are either preselected or highly variable, based on a clustering heuristic.

    Parameters
    ----------
    adata :
    n_top_genes : int
        Number of genes to keep
    preselected_genes : int, optional
        Names of genes to keep, by default None
    seed : int, optional
        Seed, by default 0
    """
    adata_ = adata.copy()
    preselected_genes = preselected_genes if preselected_genes is not None else []

    adata_log = adata_.copy()
    sc.pp.normalize_total(adata_log, target_sum=1e6)
    sc.pp.log1p(adata_log)
    pca_ = PCA(n_components=50).fit(adata_log.X)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3")

    clusters = KMeans(n_clusters=n_top_genes, random_state=seed).fit_predict(
        pca_.components_.T
    )
    adata.var.loc[:, "clusters"] = clusters

    adata.var.index.name = "index"
    selected_genes = (
        adata.var.reset_index()
        .groupby("clusters")
        .apply(lambda x: x.sort_values("variances_norm").iloc[-1]["index"])
        .values
    )
    union_genes = np.union1d(selected_genes, preselected_genes)
    return adata[:, adata.var.index.isin(union_genes)].copy()


def one_hot(x):
    return np.eye(x.max() + 1)[x]


def select_architecture(
    model_cls,
    adata,
    xy_model_kwargs_grid: dict,
    **kwargs,
):
    """Selects architecture for feature selection"""
    all_losses = pd.DataFrame()
    parameter_grid = list(ParameterGrid(xy_model_kwargs_grid))
    keys_of_interest = None
    kwargs_ = kwargs.copy()
    if "xy_model_kwargs" not in kwargs_:
        kwargs_["xy_model_kwargs"] = {}
    for grid_param in parameter_grid:
        base_config_ = kwargs_.copy()
        base_config_["xy_model_kwargs"].update(grid_param)

        crt = model_cls(adata=adata, **base_config_)
        crt.train_statistic()
        losses = crt.losses.assign(**grid_param)
        all_losses = pd.concat([all_losses, losses])
        if keys_of_interest is None:
            keys_of_interest = list(grid_param.keys())
    relevant_losses = all_losses.loc[lambda x: x.metric == "stat_val_loss"]
    best_params = (
        relevant_losses.groupby(keys_of_interest)["value"]
        .min()
        .reset_index()
        .loc[:, keys_of_interest]
        .iloc[0]
        .to_dict()
    )
    return {**kwargs, **best_params}
