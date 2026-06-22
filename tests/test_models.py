import scanpy as sc
from scvi.model import JaxSCVI
from scvi.data import synthetic_iid
from vivs._models import LinearImportanceScore, NeuralImportanceScore


def test_lvm():
    adata = synthetic_iid()
    JaxSCVI.setup_anndata(adata)
    model = JaxSCVI(adata)
    model.train(max_epochs=5)
    model.get_latent_representation()


def test_importance_score():
    adata = synthetic_iid()
    LinearImportanceScore.setup_anndata(adata, y_obsm_key="protein_expression", batch_key="batch")
    importance_score = LinearImportanceScore(adata)
    importance_score.train(max_epochs=5, accelerator="cpu")

    NeuralImportanceScore.setup_anndata(adata, y_obsm_key="protein_expression")
    importance_score = NeuralImportanceScore(adata)
    importance_score.train(max_epochs=5)
