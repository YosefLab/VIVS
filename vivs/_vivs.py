import os

import fastcluster
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import plotnine as p9
import scanpy as sc
import xarray as xr
from flax import serialization
from ml_collections import ConfigDict
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from vivs._dl_utils import construct_dataloader
from vivs._models import JAXSCVAE, ImportanceScorer, ImportanceScorerLinear
from vivs._training import train_scvi, train_statistic
from vivs._utils import one_hot

from ._constants import REGISTRY_KEYS


class VIVS:
    def __init__(
        self,
        adata: sc.AnnData,
        feature_obsm_key: str = "protein_expression",
        feature_names: str = None,
        batch_key: str = None,
        n_mc_samples: int = 500,
        percent_dev: float = 0.5,
        compute_pvalue_on: str = "val",
        **config_kwargs,
    ):
        self.n_mc_samples = n_mc_samples

        self.config = self.get_default_config()
        self.config.update(config_kwargs)

        self.n_genes = adata.X.shape[1]
        self.n_features = adata.obsm[feature_obsm_key].shape[-1]
        self.n_batch = (
            adata.obs[batch_key].unique().shape[0] if batch_key is not None else 0
        )
        feature_expression = adata.obsm[feature_obsm_key]

        if batch_key is not None:
            adata.obs.loc[:, "_batch_indices"] = (
                adata.obs[batch_key].astype("category").cat.codes.values
            )
            self.batch_key = "_batch_indices"
        else:
            self.batch_key = None

        # Preprocess target expression
        self.internal_feature_key = "_vivs_feature_expression"
        self.feature_field_name = "response"
        xy_loss = self.config["xy_model_kwargs"]["loss_type"]
        if xy_loss == "mse":
            adata.obsm[self.internal_feature_key] = (
                feature_expression - feature_expression.mean(0)
            ) / feature_expression.std(0)
        elif xy_loss == "binary":
            adata.obsm[self.internal_feature_key] = feature_expression

        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = (
                adata.obsm[feature_obsm_key].columns
                if hasattr(adata.obsm[feature_obsm_key], "columns")
                else np.arange(self.n_features)
            )
        self.compute_pvalue_on = compute_pvalue_on

        # Generative model
        self.x_tx = self.config["x_train_kwargs"]["tx"]
        self.xy_tx = self.config["xy_train_kwargs"]["tx"]
        self.x_model_kwargs = self.config["x_model_kwargs"]
        self.x_model_kwargs["n_input"] = self.n_genes

        # Feature importance model
        self.xy_include_batch = self.config["xy_include_batch_in_input"]
        self.xy_input_size = (
            self.n_genes + self.n_batch if self.xy_include_batch else self.n_genes
        )
        xy_linear = self.config["xy_linear"]
        if xy_linear:
            self.importance_cls = ImportanceScorerLinear
            self.importance_kwargs = dict(
                loss_type=self.config["xy_model_kwargs"]["loss_type"],
            )
        else:
            self.importance_cls = ImportanceScorer
            self.importance_kwargs = self.config["xy_model_kwargs"]
        self.importance_kwargs["n_features"] = self.n_features
        self.batch_size = self.config["batch_size"]
        self.n_epochs = self.config["n_epochs"]

        # Useful preprocessing
        adata_log = adata.copy()
        sc.pp.normalize_total(adata_log, target_sum=1e6)
        sc.pp.log1p(adata_log)
        self.adata_log = adata_log
        n_obs = adata.X.shape[0]
        np.random.seed(0)
        self.is_dev = np.random.random(n_obs) <= percent_dev
        self.adata = adata

        # Construct train and validation datasets
        self.adata.obs.loc[:, "is_dev"] = self.is_dev
        self.train_adata = self.adata[self.adata.obs["is_dev"]].copy()
        self.val_adata = self.adata[~self.adata.obs["is_dev"]].copy()

        # Metrics and parameters
        self.x_params = None
        self.x_batch_stats = None
        self.xy_batch_stats = None
        self.xy_params = None
        self.losses = None

    @staticmethod
    def get_default_config():
        config = ConfigDict()
        config.batch_size = 128
        config.n_epochs = 200
        config.x_model_kwargs = dict(
            n_input=None,
            n_latent=10,
            likelihood="nb",
            dropout_rate=0.0,
            n_hidden=128,
            last_h_activation="softplus",
        )
        config.x_train_kwargs = dict(
            lr=1e-3,
            patience=20,
            tx=None,
            early_stopping_metric="elbo",
            n_epochs_kl_warmup=50,
        )

        config.xy_linear = False
        config.xy_include_batch_in_input = False
        config.xy_model_kwargs = dict(
            n_features=None,
            n_hidden=128,
            dropout_rate=0.0,
            loss_type="mse",
        )

        config.xy_train_kwargs = dict(
            patience=20,
            lr=1e-3,
            tx=None,
        )
        return config.lock()

    @property
    def generative_model(self) -> nn.Module:
        return JAXSCVAE(**self.x_model_kwargs)

    @property
    def statistic_model(self) -> nn.Module:
        return self.importance_cls(**self.importance_kwargs)

    def train_scvi(self):
        """Train the generative model."""

        losses, x_params, x_batch_stats = train_scvi(
            generative_model=self.generative_model,
            train_adata=self.train_adata,
            val_adata=self.val_adata,
            batch_size=self.batch_size,
            batch_key=self.batch_key,
            protein_key=self.internal_feature_key,
            n_epochs=self.n_epochs,
            **self.config["x_train_kwargs"],
        )
        self.set_or_update_metrics(losses)
        self.x_params = x_params
        self.x_batch_stats = x_batch_stats

    def train_statistic(self):
        """Train the feature importance model."""

        losses, xy_params, xy_batch_stats = train_statistic(
            statistic_model=self.statistic_model,
            train_adata=self.train_adata,
            val_adata=self.val_adata,
            batch_size=self.batch_size,
            batch_key=self.batch_key,
            protein_key=self.internal_feature_key,
            n_epochs=self.n_epochs,
            include_batch_in_input=self.xy_include_batch,
            **self.config["xy_train_kwargs"],
        )
        self.set_or_update_metrics(losses)
        self.xy_params = xy_params
        self.xy_batch_stats = xy_batch_stats

    def train_all(self):
        self.train_scvi()
        self.train_statistic()

    def get_importance(self, eval_adata=None, batch_size=128, n_mc_per_pass=1):
        if eval_adata is None:
            if self.compute_pvalue_on == "val":
                eval_adata = self.adata[~self.adata.obs["is_dev"]].copy()
            elif self.compute_pvalue_on == "all":
                eval_adata = self.adata.copy()
        n_obs = eval_adata.X.shape[0]

        eval_dl = construct_dataloader(
            eval_adata,
            batch_size=batch_size,
            shuffle=False,
            batch_key=self.batch_key,
            protein_key=self.internal_feature_key,
        )
        total_its = n_obs // batch_size

        rng = jax.random.PRNGKey(0)
        x_rng, z_rng = jax.random.split(rng)

        tilde_ts = jnp.zeros((self.n_mc_samples, self.n_genes, self.n_features))
        observed_ts = jnp.zeros((self.n_features,))
        gene_ids = jnp.arange(self.n_genes)

        @jax.jit
        def get_tilde_t(x, batch_indices, y):
            x_ = jnp.log1p(1e6 * x / jnp.sum(x, axis=-1, keepdims=True))
            x_ = self.process_xy_input(x_, batch_indices)
            res = self.statistic_model.apply(
                {
                    "params": self.xy_params,
                    "batch_stats": self.xy_batch_stats,
                },
                x_,
                y,
                training=False,
            )["all_loss"].sum(0)
            return res

        @jax.jit
        def randomize(inputs, z_rng):
            outs = self.generative_model.apply(
                {
                    "params": self.x_params,
                    "batch_stats": self.x_batch_stats,
                },
                *inputs,
                rngs={"z": z_rng},
                training=False,
            )
            px = outs["px"]
            return px

        def _compute_loss(x, xtilde, gene_id, batch_indices, y):
            x_ = x.at[..., gene_id].set(xtilde)
            x__ = jnp.log1p(1e6 * x_ / jnp.sum(x_, axis=-1, keepdims=True))
            x__ = self.process_xy_input(x__, batch_indices)
            # shape (n_cells, n_proteins)
            res = self.statistic_model.apply(
                {
                    "params": self.xy_params,
                    "batch_stats": self.xy_batch_stats,
                },
                x__,
                y,
                training=False,
            )["all_loss"].sum(0)
            return res

        @jax.jit
        def compute_tilde_t(x, px, x_rng, batch_indices, y):
            _x_tilde = px.sample(x_rng)
            _tilde_t_k = jax.vmap(_compute_loss, (None, -1, 0, None, None), 0)(
                x, _x_tilde, gene_ids, batch_indices, y
            )
            return _tilde_t_k

        @jax.jit
        def double_compute_tilde_t(x, px, x_rng, batch_indices, y):
            _x_tilde = px.sample(x_rng, sample_shape=(n_mc_per_pass,))
            _fn = jax.vmap(_compute_loss, (None, -1, 0, None, None), 0)
            _fn = jax.vmap(_fn, (None, 0, None, None, None), 0)
            return _fn(x, _x_tilde, gene_ids, batch_indices, y)

        n_passes = self.n_mc_samples // n_mc_per_pass

        for tensors in tqdm(eval_dl, total=total_its):
            x = jnp.array(tensors[REGISTRY_KEYS.X_KEY])
            batch_indices = jnp.array(tensors[REGISTRY_KEYS.BATCH_KEY])
            protein_expression = jnp.array(tensors[self.feature_field_name])

            observed_ts += get_tilde_t(x, batch_indices, protein_expression) / n_obs
            px = randomize([x, batch_indices], z_rng)

            _tilde_t = []
            if n_mc_per_pass == 1:
                for _ in range(self.n_mc_samples):
                    _tilde_t_k = compute_tilde_t(
                        x, px, x_rng, batch_indices, protein_expression
                    )
                    _tilde_t.append(_tilde_t_k[None])
                    x_rng, _ = jax.random.split(x_rng)
            else:
                for _ in range(n_passes):
                    _tilde_t_k = double_compute_tilde_t(
                        x, px, x_rng, batch_indices, protein_expression
                    )
                    _tilde_t.append(_tilde_t_k)
                    x_rng, _ = jax.random.split(x_rng)
            _tilde_t = jnp.concatenate(_tilde_t, axis=0)
            tilde_ts += _tilde_t / n_obs
            z_rng, _ = jax.random.split(z_rng)
        pval = (1.0 + (observed_ts >= tilde_ts).sum(0)) / (1.0 + self.n_mc_samples)
        padj = np.array(
            [multipletests(_pval, method="fdr_bh")[1] for _pval in pval.T]
        ).T

        return dict(
            obs_ts=np.asarray(observed_ts),
            null_ts=np.asarray(tilde_ts),
            pvalues=np.asarray(pval),
            padj=padj,
        )

    def get_cell_scores(
        self,
        gene_ids,
        protein_ids=None,
        eval_adata=None,
        batch_size=None,
        n_mc_samples=None,
    ):
        if eval_adata is None:
            eval_adata = self.adata[~self.adata.obs["is_dev"]].copy()
        n_obs = eval_adata.X.shape[0]
        batch_size = batch_size if batch_size is not None else self.batch_size
        n_mc_samples = n_mc_samples if n_mc_samples is not None else self.n_mc_samples

        eval_dl = construct_dataloader(
            eval_adata,
            batch_size=batch_size,
            shuffle=False,
            batch_key=self.batch_key,
            protein_key=self.internal_feature_key,
        )
        total_its = n_obs // batch_size

        rng = jax.random.PRNGKey(0)
        x_rng, z_rng = jax.random.split(rng)

        gene_ids = jnp.array(gene_ids)
        protein_ids = (
            jnp.array(protein_ids)
            if protein_ids is not None
            else jnp.arange(self.n_features)
        )
        tilde_t_mean = []
        obs_t = []

        @jax.jit
        def get_tilde_t_nosum(x, batch_indices, y):
            x_ = jnp.log1p(1e6 * x / jnp.sum(x, axis=-1, keepdims=True))
            x_ = self.process_xy_input(x_, batch_indices)
            res = self.statistic_model.apply(
                {
                    "params": self.xy_params,
                    "batch_stats": self.xy_batch_stats,
                },
                x_,
                y,
                training=False,
            )["all_loss"][..., protein_ids]
            return res

        def _compute_loss(x, xtilde, gene_id, batch_indices, y):
            x_ = x.at[..., gene_id].set(xtilde)
            x__ = jnp.log1p(1e6 * x_ / jnp.sum(x_, axis=-1, keepdims=True))
            x__ = self.process_xy_input(x__, batch_indices)
            # shape (n_cells, n_proteins)
            res = self.statistic_model.apply(
                {
                    "params": self.xy_params,
                    "batch_stats": self.xy_batch_stats,
                },
                x__,
                y,
                training=False,
            )["all_loss"][..., protein_ids]
            return res

        @jax.jit
        def randomize(inputs, z_rng):
            outs = self.generative_model.apply(
                {
                    "params": self.x_params,
                    "batch_stats": self.x_batch_stats,
                },
                *inputs,
                rngs={"z": z_rng},
                training=False,
            )
            px = outs["px"]
            return px

        @jax.jit
        def compute_tilde_t(x, px, x_rng, batch_indices, y):
            _x_tilde = px.sample(x_rng)[..., gene_ids]
            _tilde_t_k = jax.vmap(_compute_loss, (None, -1, 0, None, None), 1)(
                x, _x_tilde, gene_ids, batch_indices, y
            )
            return _tilde_t_k

        for tensors in tqdm(eval_dl, total=total_its):
            x = jnp.array(tensors[REGISTRY_KEYS.X_KEY])
            batch_indices = jnp.array(tensors[REGISTRY_KEYS.BATCH_KEY])
            protein_expression = jnp.array(tensors[self.feature_field_name])
            px = randomize([x, batch_indices], z_rng)
            _tilde_t = []
            for _ in range(n_mc_samples):
                _tilde_t_k = compute_tilde_t(
                    x, px, x_rng, batch_indices, protein_expression
                )
                # _tilde_t = _tilde_t.at[mc_sample].set(_tilde_t_k)
                _tilde_t.append(_tilde_t_k[None])
                x_rng, _ = jax.random.split(x_rng)
            _tilde_t = jnp.concatenate(_tilde_t, axis=0).mean(0)
            observed_ts = get_tilde_t_nosum(x, batch_indices, protein_expression)

            z_rng, _ = jax.random.split(z_rng)
            # score = _tilde_t - observed_ts[:, None]
            tilde_t_mean.append(np.asarray(_tilde_t))
            obs_t.append(np.asarray(observed_ts[:, None]))
        tilde_t_mean = np.concatenate(tilde_t_mean)
        obs_t = np.concatenate(obs_t)
        return dict(
            tilde_t_mean=tilde_t_mean,
            obs_t=obs_t,
        )

    def save_params(self, save_path, save_adata=False):
        x_params_bytes = serialization.to_bytes(self.x_params)
        x_batch_stats_bytes = serialization.to_bytes(self.x_batch_stats)
        xy_batch_stats_bytes = serialization.to_bytes(self.xy_batch_stats)
        xy_params_bytes = serialization.to_bytes(self.xy_params)

        files = [
            (x_params_bytes, "x_params_bytes"),
            (x_batch_stats_bytes, "x_batch_stats_bytes"),
            (xy_batch_stats_bytes, "xy_batch_stats_bytes"),
            (xy_params_bytes, "xy_params_bytes"),
        ]
        for file, name in files:
            with open(os.path.join(save_path, name), "wb") as f:
                f.write(file)

        if save_adata:
            self.adata.write(os.path.join(save_path, "adata.h5ad"))

        self.init_params.to_csv(os.path.join(save_path, "init_params.csv"))

    def get_gene_correlations(self, adata=None):
        """Compute G times G gene correlation matrix."""

        cpu_device = jax.devices("cpu")[0]
        scdl = construct_dataloader(
            adata,
            batch_size=self.batch_size,
            shuffle=True,
            batch_key=self.batch_key,
            protein_key=self.internal_feature_key,
        )

        @jax.jit
        def get_scales(inputs, z_rng):
            return self.generative_model.apply(
                {
                    "params": self.x_params,
                    "batch_stats": self.x_batch_stats,
                },
                *inputs,
                rngs={"z": z_rng},
                training=False,
            )["h"]

        xx_est = jax.device_put(
            jnp.zeros((self.n_genes, self.n_genes)), device=cpu_device
        )
        x_est = jax.device_put(jnp.zeros(self.n_genes), device=cpu_device)
        z_rng = jax.random.PRNGKey(0)
        n_obs_total = adata.X.shape[0]
        for tensors in scdl:
            x = jnp.array(tensors[REGISTRY_KEYS.X_KEY])
            batch_indices = jnp.array(tensors[REGISTRY_KEYS.BATCH_KEY])
            z_rng = jax.random.PRNGKey(0)
            z_rng, _ = jax.random.split(z_rng)
            scales = get_scales([x, batch_indices], z_rng)
            x_est += jax.device_put(
                jnp.sum(scales, axis=0) / n_obs_total, device=cpu_device
            )
            xx_est += jax.device_put(
                jnp.sum(
                    jax.lax.batch_matmul(scales[..., None], scales[:, None]), axis=0
                )
                / n_obs_total,
                device=cpu_device,
            )
        x_est = x_est[None]
        cov_ = xx_est - jnp.matmul(x_est.T, x_est)
        factor_ = 1.0 / jnp.sqrt(jnp.diag(cov_))
        dmat = jnp.diag(factor_)
        corr_ = dmat @ (cov_ @ dmat)
        return np.asarray(corr_)

    def get_gene_groupings(
        self,
        adata=None,
        method="complete",
        return_z=False,
        n_clusters_list=None,
    ):
        """Computes gene groupings based on gene correlations.

        Parameters
        ----------
        adata :
            adata used to compute gene correlations.
        method :
            Linkage for hierarchical clustering.
        return_z :
            Whether to return linkage matrix.
        n_clusters_list :
            Number of desired clusters.
        """
        adata = self.adata[self.adata.obs.is_dev].copy() if adata is None else adata
        corr_ = self.get_gene_correlations(adata=adata)
        pseudo_dist = 1 - corr_
        pseudo_dist = (pseudo_dist + pseudo_dist.T) / 2
        pseudo_dist = np.clip(pseudo_dist, a_min=0.0, a_max=100.0)
        pseudo_dist = pseudo_dist - np.diag(np.diag(pseudo_dist))
        dist_vec = squareform(pseudo_dist, checks=False)
        Z = fastcluster.linkage(dist_vec, method=method)
        Z = hierarchy.optimal_leaf_ordering(Z, dist_vec)
        gene_order = hierarchy.leaves_list(Z)
        gene_order = self.adata.var_names[gene_order].values

        assert n_clusters_list is not None
        if not isinstance(n_clusters_list, list):
            n_clusters_list = [n_clusters_list]
        gene_groupings = []
        for n_cluster in n_clusters_list:
            if n_cluster >= self.n_genes:
                continue
            cluster_assignments = hierarchy.fcluster(Z, n_cluster, criterion="maxclust")
            cluster_assignments -= 1
            gene_groupings.append(cluster_assignments)
        if return_z:
            return (
                gene_groupings,
                Z,
                gene_order,
            )
        return gene_groupings

    def predict_t(self, adata=None, batch_size=128):
        @jax.jit
        def get_t(x, y):
            x_ = jnp.log1p(1e6 * x / jnp.sum(x, axis=-1, keepdims=True))
            res = self.statistic_model.apply(
                {
                    "params": self.xy_params,
                    "batch_stats": self.xy_batch_stats,
                },
                x_,
                y,
                training=False,
            )["all_loss"]
            return res

        adata = self.adata if adata is None else adata
        eval_dl = construct_dataloader(
            adata,
            batch_size=batch_size,
            shuffle=False,
            batch_key=self.batch_key,
            protein_key=self.internal_feature_key,
        )
        res = []
        for tensors in eval_dl:
            x = jnp.array(tensors[REGISTRY_KEYS.X_KEY])
            # batch_indices = jnp.array(tensors[REGISTRY_KEYS.BATCH_KEY])
            protein_expression = jnp.array(tensors[self.feature_field_name])
            res.append(np.array(get_t(x, protein_expression)))
        return np.concatenate(res, axis=0)

    def get_latent(self):
        dl = construct_dataloader(
            self.adata,
            batch_size=128,
            shuffle=False,
            batch_key=self.batch_key,
            protein_key=self.internal_feature_key,
        )

        @jax.jit
        def _get_latent(inputs, z_rng):
            outs = self.generative_model.apply(
                {
                    "params": self.x_params,
                    "batch_stats": self.x_batch_stats,
                },
                *inputs,
                rngs={"z": z_rng},
                training=False,
            )
            return outs["qz"].loc

        cpu_device = jax.devices("cpu")[0]
        zs = []
        z_rng = jax.random.PRNGKey(0)
        for tensors in dl:
            x = jnp.array(tensors[REGISTRY_KEYS.X_KEY])
            batch_indices = jnp.array(tensors[REGISTRY_KEYS.BATCH_KEY])
            qz = _get_latent([x, batch_indices], z_rng)
            qz = jax.device_put(qz, device=cpu_device)
            zs.append(qz)
        zs = jnp.concatenate(zs, axis=0)
        return np.array(zs)

    def get_hier_importance(
        self,
        n_clusters_list: list,
        eval_adata: sc.AnnData = None,
        batch_size=128,
        gene_groupings=None,
        gene_order=None,
        clustering_method="complete",
        use_vmap=True,
    ):
        """
        Performs hierarchical importance testing.

        This method first computes groups of genes based on their correlations, or alternatively,
        relies on pre-computed gene groupings at several resolutions.
        At each resolution, conditional independence testing is performed using the CRT.

        Once this method has been called, the results can be visualized using the `plot_hier_importance` method.

        This method returns three objects:
        - `gene_results`: an xarray object containing significance scores for each gene and protein.
        - `cluster_results` and `gene_to_cluster`: dataframes that are mostly useful for visualization,
        and can be ignored in most cases.

        Parameters
        ----------
        cluster_sizes
            List of cluster sizes, defining the resolutions at which to perform conditional independence testing.
            For instance, if cluster_sizes = [5, 10], then the method will
            test for conditional independence based on a partition of the genes into 5, 10 clusters, as well as all genes.
        eval_adata
            Anndata object to evaluate on. If None, then the validation set is used.
        batch_size
            Batch size for evaluation.
        gene_groupings
            Optional gene groupings. If None, then gene groupings are computed using hierarchical clustering.
        gene_order
            Optional gene order. This only determines how the genes are ordered in the visualization.
            If None, then gene order is computed using hierarchical clustering.
        clustering_method
            Linkage method for hierarchical clustering.
        use_vmap
            Whether to use vmap for parallelization.
        """
        if eval_adata is None:
            if self.compute_pvalue_on == "val":
                eval_adata = self.adata[~self.adata.obs["is_dev"]].copy()
            elif self.compute_pvalue_on == "all":
                eval_adata = self.adata.copy()
        n_obs = eval_adata.X.shape[0]

        # construct dataloader
        eval_dl = construct_dataloader(
            eval_adata,
            batch_size=batch_size,
            shuffle=False,
            batch_key=self.batch_key,
            protein_key=self.internal_feature_key,
        )
        total_its = n_obs // batch_size

        # compute gene groups
        train_adata = self.adata[self.adata.obs["is_dev"]].copy()
        if gene_groupings is None:
            gene_groupings, _, gene_order = self.get_gene_groupings(
                adata=train_adata,
                n_clusters_list=n_clusters_list,
                return_z=True,
                method=clustering_method,
            )
        else:
            if gene_order is None:
                gene_order = np.arange(self.n_genes)
        gene_groupings.append(np.arange(self.n_genes).astype(np.int32))
        gene_groups_oh = [jnp.array(one_hot(grouping)) for grouping in gene_groupings]
        gene_groups_sizes = [gg.shape[1] for gg in gene_groups_oh]

        rng = jax.random.PRNGKey(0)
        x_rng, z_rng = jax.random.split(rng)

        tilde_ts = [
            jnp.zeros((self.n_mc_samples, sz, self.n_features))
            for sz in gene_groups_sizes
        ]
        observed_ts = jnp.zeros((self.n_features,))

        @jax.jit
        def get_tilde_t(x, y):
            x_ = jnp.log1p(1e6 * x / jnp.sum(x, axis=-1, keepdims=True))
            res = self.statistic_model.apply(
                {
                    "params": self.xy_params,
                    "batch_stats": self.xy_batch_stats,
                },
                x_,
                y,
                training=False,
            )["all_loss"].sum(0)
            return res

        def randomize(inputs, z_rng):
            outs = self.generative_model.apply(
                {
                    "params": self.x_params,
                    "batch_stats": self.x_batch_stats,
                },
                *inputs,
                rngs={"z": z_rng},
                training=False,
            )
            px = outs["px"]
            return px

        def _compute_loss_group(x, xtilde, gene_groups, y):
            x_ = x * (1.0 - gene_groups) + (xtilde * gene_groups)
            x__ = jnp.log1p(1e6 * x_ / jnp.sum(x_, axis=-1, keepdims=True))
            # shape (n_cells, n_proteins)
            res = self.statistic_model.apply(
                {
                    "params": self.xy_params,
                    "batch_stats": self.xy_batch_stats,
                },
                x__,
                y,
                training=False,
            )["all_loss"].sum(0)
            return res

        @jax.jit
        def compute_tilde_t_group(x, x_tilde, y, gene_groups):
            if use_vmap:
                _tilde_t_k = jax.vmap(_compute_loss_group, (None, None, -1, None), 0)(
                    x, x_tilde, gene_groups, y
                )
            else:

                def parallel_fn(gene_group):
                    return _compute_loss_group(x, x_tilde, gene_group, y)

                _tilde_t_k = jax.lax.map(parallel_fn, gene_groups)
            return _tilde_t_k

        for tensors in tqdm(eval_dl, total=total_its):
            x = jnp.array(tensors[REGISTRY_KEYS.X_KEY])
            batch_indices = jnp.array(tensors[REGISTRY_KEYS.BATCH_KEY])
            protein_expression = jnp.array(tensors[self.feature_field_name])

            observed_ts += get_tilde_t(x, protein_expression) / n_obs
            px = randomize([x, batch_indices], z_rng)

            for k in range(self.n_mc_samples):
                x_tilde = px.sample(x_rng)
                x_rng, _ = jax.random.split(x_rng)
                for gene_group_idx, gene_group_oh in enumerate(gene_groups_oh):
                    _tilde_t_k = (
                        compute_tilde_t_group(
                            x, x_tilde, protein_expression, gene_group_oh
                        )
                        / n_obs
                    )

                    updated_tilde_t = tilde_ts[gene_group_idx][k] + _tilde_t_k
                    tilde_ts[gene_group_idx] = (
                        tilde_ts[gene_group_idx].at[k].set(updated_tilde_t)
                    )
            z_rng, _ = jax.random.split(z_rng)

        gene_results = self._construct_results(
            observed_ts=observed_ts,
            tilde_ts=tilde_ts,
            gene_groupings=gene_groupings,
            gene_groups_sizes=gene_groups_sizes,
            gene_order=gene_order,
        )
        return gene_results.squeeze()

    @staticmethod
    def plot_hier_importance(
        gene_results,
        base_resolution=None,
        significance_threshold=0.1,
        plot_fig=True,
        theme_kwargs=None,
    ):
        if base_resolution is None:
            base_resolution = gene_results.resolution.values.min()
        padjs = gene_results.loc[{"resolution": base_resolution}]["padj"].to_pandas()
        genes_to_plot = padjs.loc[lambda x: x < significance_threshold].index.tolist()
        res_subset = gene_results.loc[{"gene_name": genes_to_plot}].assign(
            gene_index=("gene_name", np.arange(len(genes_to_plot)))
        )
        plot_df = []
        for resolution_idx, resolution in enumerate(res_subset.resolution.values):
            res_ = res_subset.loc[{"resolution": resolution}]
            unique_clusters = np.unique(res_["cluster_assignment"].values)
            for cluster in unique_clusters:
                gene_is_in_cluster = res_["cluster_assignment"] == cluster
                res_cluster = res_.loc[{"gene_name": gene_is_in_cluster}]
                xmin = res_cluster.gene_index.values.min()
                xmax = res_cluster.gene_index.values.max()
                are_indices_contiguous = (
                    xmax - xmin + 1 == res_cluster.gene_name.shape[0]
                )
                if not are_indices_contiguous:
                    raise ValueError("Gene indices are not contiguous")
                is_cluster_detected = (res_cluster["padj"] < 0.1).all()
                if is_cluster_detected:
                    plot_df.append(
                        {
                            "resolution_idx": resolution_idx,
                            "resolution": resolution,
                            "xmin": xmin - 0.5,
                            "xmax": xmax + 0.5,
                            "ymin": resolution_idx,
                            "ymax": resolution_idx + 1,
                        }
                    )
        plot_df = pd.DataFrame(plot_df)
        labels = list(res_subset.gene_name.values)
        breaks = list(res_subset.gene_index.values)
        if plot_fig:
            if theme_kwargs is None:
                theme_kwargs = dict(figure_size=(15, 2))
            fig = (
                p9.ggplot(plot_df)
                + p9.geom_rect(
                    plot_df,
                    p9.aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
                    inherit_aes=False,
                    fill="#ededed",
                    color="#080808",
                )
                + p9.scale_x_continuous(labels=labels, breaks=breaks)
                + p9.theme_classic()
                + p9.theme(
                    axis_text_x=p9.element_text(rotation=90),
                    axis_line_y=p9.element_blank(),
                    axis_text_y=p9.element_blank(),
                    axis_ticks_major_y=p9.element_blank(),
                )
                + p9.labs(x="", y="")
                + p9.theme(**theme_kwargs)
            )
            return fig
        else:
            return plot_df, labels, breaks

    def _construct_results(
        self,
        observed_ts: jnp.ndarray,
        tilde_ts: jnp.ndarray,
        gene_groupings: list,
        gene_groups_sizes: list,
        gene_order: list,
    ):
        """
        Constructs the results for the hierarchical importance model based on the
        null and randomized statistics and the gene groupings.

        Parameters
        ----------
        observed_ts
            The observed statistics.
        tilde_ts
            The randomized statistics.
        gene_groupings
            The gene groupings, of shape (n_resolutions, n_genes).
        gene_groups_sizes
            The total number of gene groups per resolution, of shape (n_resolutions,).
        """
        datasets = []
        for resolution_idx, resolution in enumerate(gene_groups_sizes):
            pvals = (1.0 + (observed_ts >= tilde_ts[resolution_idx]).sum(0)) / (
                1.0 + self.n_mc_samples
            )

            padjs = [
                multipletests(pvals[:, protein_id], method="fdr_bh")[1]
                for protein_id in range(self.n_features)
            ]
            padjs = jnp.stack(padjs, axis=-1)
            gene_clusters = gene_groupings[resolution_idx]
            pvals_ = pvals[gene_clusters]
            padjs_ = padjs[gene_clusters]
            coords_ = dict(
                dims=["gene_name", "feature"],
                coords={
                    "gene_name": self.adata.var_names.values,
                    "feature": self.feature_names,
                    "resolution": resolution,
                    "resolution_idx": resolution_idx,
                },
            )
            pvals_ = xr.DataArray(pvals_, **coords_)
            padjs_ = xr.DataArray(padjs_, **coords_)
            cluster_assignments = xr.DataArray(
                gene_clusters,
                dims=["gene_name"],
                coords={
                    "gene_name": self.adata.var_names.values,
                    "resolution": resolution,
                    "resolution_idx": resolution_idx,
                },
            )
            datasets.append(
                xr.Dataset(
                    {
                        "pval": pvals_,
                        "padj": padjs_,
                        "cluster_assignment": cluster_assignments,
                    }
                )
            )
        return xr.concat(datasets, dim="resolution").reindex(gene_name=gene_order)

    def set_or_update_metrics(self, metrics: pd.DataFrame):
        """
        Update the metrics dataframe.
        """

        if self.losses is None:
            self.losses = metrics
        else:
            self.losses = pd.concat([self.losses, metrics])

    def process_xy_input(
        self, x: jnp.ndarray, batch_indices: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Process inputs for the feature importance model.
        """
        return (
            jnp.concatenate(
                [x, jax.nn.one_hot(batch_indices.squeeze(-1), self.n_batch)], axis=-1
            )
            if self.xy_include_batch
            else x
        )
