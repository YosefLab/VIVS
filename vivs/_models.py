from vivs._constants import REGISTRY_KEYS
from scvi.model.base import BaseModelClass, JaxTrainingMixin
from scvi.data import AnnDataManager, fields
from scvi.utils import setup_anndata_dsp

from abc import ABC


from ._modules import NeuralNet, LinearModel


class _ImportanceScore(ABC, JaxTrainingMixin, BaseModelClass):
    """
    Abstract base class for Importance Score models.

    This class provides the common structure and `setup_anndata` method for
    various Importance Score implementations. It is not intended for direct
    instantiation. Instead, use concrete subclasses like `LinearImportanceScore`
    or `NeuralImportanceScore`.
    """

    def __init__(self, adata):
        super().__init__(adata)
        self.n_features = self.summary_stats.n_vars
        self.n_responses = self.summary_stats.n_Y

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata,
        y_obsm_key: str,
        y_names_uns_key: str | None = None,
        batch_key: str | None = None,
        layer: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        protein_expression_obsm_key
            key in `adata.obsm` for protein expression data.
        protein_names_uns_key
            key in `adata.uns` for protein names. If None, will use the column names of
            `adata.obsm[protein_expression_obsm_key]` if it is a DataFrame, else will assign
            sequential names to proteins.
        %(param_batch_key)s
        %(param_layer)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s

        Returns
        -------
        %(returns)s
        """

        setup_method_args = cls._get_setup_method_args(**locals())
        batch_field = fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key)
        anndata_fields = [
            fields.LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            fields.CategoricalObsField(
                REGISTRY_KEYS.LABELS_KEY, None
            ),  # Default labels field for compatibility with TOTALVAE
            fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            fields.NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            fields.CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            fields.NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
            fields.ProteinObsmField(
                REGISTRY_KEYS.Y_KEY,
                y_obsm_key,
                use_batch_mask=True,
                batch_field=batch_field,
                colnames_uns_key=y_names_uns_key,
                is_count_data=False,
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def to_device(self, device):
        pass

    @property
    def device(self):
        return self.module.device


class LinearImportanceScore(_ImportanceScore):
    def __init__(self, adata):
        super().__init__(adata)
        n_batch = self.summary_stats.n_batch
        self.module = LinearModel(
            n_features=self.n_features, n_batch=n_batch, loss_type="mse"
        )


class NeuralImportanceScore(_ImportanceScore):
    def __init__(
        self,
        adata,
        n_hidden: int = 100,
        dropout_rate: float = 0.1,
        loss_type: str = "mse",
    ):
        super().__init__(adata)
        n_batch = self.summary_stats.n_batch
        self.module = NeuralNet(
            n_features=self.n_features,
            n_batch=n_batch,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            loss_type="mse",
        )
