import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from torch.utils.data import DataLoader, Dataset

from ._constants import REGISTRY_KEYS


class AnnDataset(Dataset):
    def __init__(
        self,
        adata: AnnData,
        response_key: str,
        batch_key: str = None,
    ):
        self.adata = adata

        self.x_mat = adata.X
        self.is_sparse = sp.issparse(self.x_mat)
        if batch_key is None:
            self.batch_indices = np.zeros((adata.n_obs, 1))
        else:
            # Batch indices should already be integers
            batch_indices = adata.obs[batch_key].values
            self.batch_indices = batch_indices[:, None]

        self.responses = adata.obsm[response_key]

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        xmat = self.x_mat[idx]
        if sp.issparse(xmat):
            xmat = xmat.toarray()
        return {
            REGISTRY_KEYS.X_KEY: xmat,
            REGISTRY_KEYS.BATCH_KEY: self.batch_indices[idx],
            "response": self.responses[idx],
        }


def construct_dataloader(
    adata: AnnData,
    batch_size: int = 128,
    shuffle: bool = False,
    batch_key: str = None,
    protein_key: str = None,
):
    # Temporary fix while scvi-tools doesn't support jax
    dataset = AnnDataset(
        adata,
        response_key=protein_key,
        batch_key=batch_key,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
