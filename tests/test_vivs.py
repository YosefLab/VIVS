import numpy as np
import pandas as pd
import scanpy as sc

from vivs import select_architecture, select_genes
from vivs._vivs import VIVS


def generate_random_data(n_cells, n_genes, n_proteins):
    X = np.random.randint(0, 300, (n_cells, n_genes))
    obs = pd.DataFrame(index=[f"cell_{idx}" for idx in range(n_cells)])
    var = pd.DataFrame(index=[f"gene_{idx}" for idx in range(n_genes)])
    protein_exp = np.random.randint(0, 100, (n_cells, n_proteins))
    return sc.AnnData(
        X=X,
        obs=obs,
        var=var,
        obsm={"protein_expression": protein_exp},
    )


def test_selection():
    adata = generate_random_data(1000, 300, 10)
    adata_ = select_genes(adata, n_top_genes=50)
    assert adata_.shape == (1000, 50)


def test_jax():
    adata = generate_random_data(300, 25, 10)
    adata.var_names = [f"gene_{idx}" for idx in range(adata.shape[1])]

    crttool = VIVS(
        adata,
        n_epochs=1,
        percent_dev=0.5,
        n_mc_samples=20,
        xy_linear=False,
    )
    crttool.train_all()
    crttool.get_importance()
    crttool.get_importance(n_mc_per_pass=10)
    crttool.get_hier_importance(n_clusters_list=[5, 10])
    crttool.get_cell_scores(gene_ids=[1, 2, 3])
    crttool.get_latent()

    crttool = VIVS(
        adata,
        n_epochs=1,
        percent_dev=0.5,
        n_mc_samples=20,
        xy_linear=False,
        x_model_kwargs=dict(last_h_activation="softplus"),
    )
    crttool.train_all()
    crttool.get_importance()
    crttool.get_importance(n_mc_per_pass=10)
    crttool.get_hier_importance(n_clusters_list=[5, 10])
    crttool.get_cell_scores(gene_ids=[1, 2, 3])
    crttool.get_latent()

    adata.obs.loc[:, "batch_indices"] = np.random.randint(0, 2, adata.shape[0])

    other_kwargs = dict(
        xy_linear=True,
        xy_include_batch_in_input=True,
        x_model_kwargs=dict(likelihood="nb"),
    )
    crttool = VIVS(
        adata,
        batch_key="batch_indices",
        n_epochs=1,
        percent_dev=0.5,
        n_mc_samples=20,
        **other_kwargs,
    )
    crttool.train_all()
    crttool.get_importance()
    crttool.get_importance(n_mc_per_pass=10)
    crttool.get_cell_scores(gene_ids=[1, 2, 3])
    crttool.get_latent()

    adata.obsm["protein_expression"] = np.random.rand(adata.shape[0], 1) >= 0.5
    other_kwargs = dict(
        xy_linear=False,
        x_model_kwargs=dict(likelihood="poisson"),
        xy_model_kwargs=dict(loss_type="binary"),
    )
    crttool = VIVS(
        adata,
        batch_key="batch_indices",
        n_epochs=1,
        percent_dev=0.5,
        n_mc_samples=20,
        **other_kwargs,
    )
    crttool.train_all()
    crttool.get_importance()
    crttool.get_importance(n_mc_per_pass=10)
    crttool.get_cell_scores(gene_ids=[1, 2, 3])

    res = crttool.get_hier_importance(n_clusters_list=[5, 10])
    crttool.plot_hier_importance(
        res, plot_fig=True, theme_kwargs=dict(figure_size=(15, 2))
    )
    crttool.plot_hier_importance(res, plot_fig=False, base_resolution=10)


def test_select_architecture():
    adata = generate_random_data(300, 25, 10)
    kwargs = dict(
        n_mc_samples=20,
        percent_dev=0.5,
    )
    grid = dict(
        n_hidden=[8, 16],
    )
    select_architecture(
        VIVS,
        adata,
        xy_model_kwargs_grid=grid,
        **kwargs,
    )
