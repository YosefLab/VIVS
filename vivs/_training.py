from copy import copy

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax.training import train_state
from tqdm import tqdm

from vivs._dl_utils import construct_dataloader

from ._constants import REGISTRY_KEYS

INCH_TO_CM = 1 / 2.54


def init_generative_model(
    generative_model,
    batch_size,
    n_genes,
):
    """Initializes the generative model."""

    init_rngs = {
        "params": jax.random.PRNGKey(0),
        "z": jax.random.PRNGKey(1),
        "dropout": jax.random.PRNGKey(2),
    }
    init_counts = jnp.ones((batch_size, n_genes))
    init_batches = jnp.zeros((batch_size, 1))

    return generative_model.init(
        init_rngs,
        init_counts,
        init_batches,
    )


def train_scvi(
    generative_model,
    train_adata,
    val_adata,
    batch_size,
    batch_key,
    protein_key,
    lr,
    n_epochs,
    n_epochs_kl_warmup,
    early_stopping_metric,
    patience,
    tx=None,
):
    """Trains the generative model."""

    # Construct dataloader
    n_genes = train_adata.shape[1]
    tx = optax.adam(lr) if tx is None else tx

    train_dl = construct_dataloader(
        train_adata,
        batch_size=batch_size,
        shuffle=True,
        batch_key=batch_key,
        protein_key=protein_key,
    )
    val_dl = construct_dataloader(
        val_adata,
        batch_size=batch_size,
        shuffle=False,
        batch_key=batch_key,
        protein_key=protein_key,
    )

    # Init state
    z_rng = jax.random.PRNGKey(0)
    d_rng = jax.random.PRNGKey(1)

    variables = init_generative_model(
        generative_model,
        batch_size=batch_size,
        n_genes=n_genes,
    )
    params = variables["params"]
    batch_stats = variables["batch_stats"]

    state = train_state.TrainState.create(
        apply_fn=generative_model.apply,
        params=params,
        tx=tx,
    )

    @jax.jit
    def train_step(state, batch_stats, inputs, z_rng, d_rng, kl_weight):
        def loss_fn(params):
            loss, variables = generative_model.apply(
                {"params": params, "batch_stats": batch_stats},
                *inputs,
                rngs={"z": z_rng, "dropout": d_rng},
                mutable=["batch_stats"],
                kl_weight=kl_weight,
                training=True,
            )
            loss = loss["loss"]
            return loss, variables

        (loss, batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        batch_stats = batch_stats["batch_stats"]
        return loss, state.apply_gradients(grads=grads), batch_stats

    @jax.jit
    def evaluate(state, batch_stats, inputs, z_rng):
        outs = generative_model.apply(
            {"params": state.params, "batch_stats": batch_stats},
            *inputs,
            rngs={"z": z_rng},
            training=False,
        )
        reco_loss = outs["reconstruction_loss"]
        elbo = outs["loss"]
        return reco_loss, elbo

    best_val_loss = np.inf
    best_x_params = copy(state.params)
    best_x_batch_stats = copy(batch_stats)
    hasnt_improved = 0
    losses = []
    for epoch in tqdm(range(n_epochs)):
        kl_weight = (
            min(1.0, epoch / n_epochs_kl_warmup) if n_epochs_kl_warmup > 0 else 1.0
        )
        for batch in train_dl:
            X = batch[REGISTRY_KEYS.X_KEY]
            batch_indices = batch[REGISTRY_KEYS.BATCH_KEY]
            X = jnp.array(X)
            batch_indices = jnp.array(batch_indices)

            z_rng, _ = jax.random.split(z_rng)
            d_rng, _ = jax.random.split(d_rng)
            loss, state, batch_stats = train_step(
                state,
                batch_stats,
                [X, batch_indices],
                z_rng,
                d_rng,
                kl_weight=kl_weight,
            )

        val_reco = 0
        val_elbo = 0
        n_batches = 0
        for batch in val_dl:
            X = batch[REGISTRY_KEYS.X_KEY]
            batch_indices = batch[REGISTRY_KEYS.BATCH_KEY]
            X = jnp.array(X)
            batch_indices = jnp.array(batch_indices)
            z_rng, _ = jax.random.split(z_rng)
            _val_reco, _val_elbo = evaluate(
                state, batch_stats, [X, batch_indices], z_rng
            )
            n_batches += 1
            val_reco += _val_reco
            val_elbo += _val_elbo

        losses.append(
            dict(
                metric="train_loss",
                value=loss.item(),
                iterate=epoch,
            )
        )
        losses.append(
            dict(
                metric="kl_weight",
                value=kl_weight,
                iterate=epoch,
            )
        )
        losses.append(
            dict(
                metric="val_reco",
                value=val_reco.item() / n_batches,
                iterate=epoch,
            )
        )
        losses.append(
            dict(
                metric="val_elbo",
                value=val_elbo.item() / n_batches,
                iterate=epoch,
            )
        )

        curr_loss = val_elbo if early_stopping_metric == "elbo" else val_reco
        if curr_loss < best_val_loss:
            best_val_loss = curr_loss
            hasnt_improved = 0
            best_x_params = copy(state.params)
            best_x_batch_stats = copy(batch_stats)
        else:
            hasnt_improved += 1
        if hasnt_improved > patience:
            break

    losses = pd.DataFrame(losses)
    return losses, best_x_params, best_x_batch_stats


def init_statistic_model(statistic_model, batch_size, n_genes, n_proteins):
    """Initializes the feature importance model."""

    init_rngs = {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)}
    init_log_counts = jnp.ones((batch_size, n_genes))
    init_prots = jnp.ones((batch_size, n_proteins))
    return statistic_model.init(init_rngs, init_log_counts, init_prots)


def train_statistic(
    statistic_model,
    train_adata,
    val_adata,
    batch_size,
    batch_key,
    protein_key,
    lr,
    n_epochs,
    patience,
    tx=None,
    normalize_x=True,
    include_batch_in_input=False,
):
    """Trains the feature importance model."""

    params = None
    batch_stats = None
    n_genes = train_adata.shape[1]
    n_proteins = train_adata.obsm[protein_key].shape[1]
    n_samples = 0 if batch_key is None else train_adata.obs[batch_key].nunique()

    tx = optax.adam(lr) if tx is None else tx

    train_dl = construct_dataloader(
        train_adata,
        batch_size=batch_size,
        shuffle=True,
        batch_key=batch_key,
        protein_key=protein_key,
    )
    val_dl = construct_dataloader(
        val_adata,
        batch_size=batch_size,
        shuffle=False,
        batch_key=batch_key,
        protein_key=protein_key,
    )

    d_rng = jax.random.PRNGKey(1)
    n_features = n_genes + n_samples if include_batch_in_input else n_genes
    variables = init_statistic_model(
        statistic_model, batch_size, n_features, n_proteins
    )
    params = variables["params"]
    batch_stats = variables["batch_stats"]

    state = train_state.TrainState.create(
        apply_fn=statistic_model.apply,
        params=params,
        tx=tx,
    )

    @jax.jit
    def train_step(state, batch_stats, inputs, dropout_rng):
        def loss_fn(params, batch_stats):
            x, batch_sample, y = inputs
            x_ = (
                jnp.log1p(1e6 * x / jnp.sum(x, axis=-1, keepdims=True))
                if normalize_x
                else x
            )
            x_ = (
                jnp.concatenate(
                    [x_, jax.nn.one_hot(batch_sample.squeeze(-1), n_samples)], axis=-1
                )
                if include_batch_in_input
                else x_
            )
            loss, variables = statistic_model.apply(
                {"params": params, "batch_stats": batch_stats},
                x_,
                y,
                rngs={"dropout": dropout_rng},
                mutable=["batch_stats"],
                training=True,
            )
            loss = loss["loss"]
            batch_stats = variables["batch_stats"]
            return loss, batch_stats

        (loss, batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, batch_stats
        )
        return loss, state.apply_gradients(grads=grads), batch_stats

    @jax.jit
    def evaluate(state, batch_stats, inputs):
        x, batch_sample, y = inputs
        x_ = (
            jnp.log1p(1e6 * x / jnp.sum(x, axis=-1, keepdims=True))
            if normalize_x
            else x
        )
        x_ = (
            jnp.concatenate(
                [x_, jax.nn.one_hot(batch_sample.squeeze(-1), n_samples)], axis=-1
            )
            if include_batch_in_input
            else x_
        )
        loss = statistic_model.apply(
            {"params": state.params, "batch_stats": batch_stats},
            x_,
            y,
            training=False,
        )["loss"]
        return loss

    best_val_loss = np.inf
    best_params = copy(state.params)
    best_batch_stats = copy(batch_stats)
    hasnt_improved = 0
    losses = []
    for epoch in tqdm(range(n_epochs)):
        for batch in train_dl:
            X = batch[REGISTRY_KEYS.X_KEY]
            protein_exp = batch["response"]
            sample_batch = batch[REGISTRY_KEYS.BATCH_KEY]
            X = jnp.array(X)
            protein_exp = jnp.array(protein_exp)
            sample_batch = jnp.array(sample_batch)

            d_rng, _ = jax.random.split(d_rng)
            loss, state, batch_stats = train_step(
                state, batch_stats, [X, sample_batch, protein_exp], d_rng
            )

        val_loss = 0
        n_batches = 0
        for batch in val_dl:
            X = batch[REGISTRY_KEYS.X_KEY]
            protein_exp = batch["response"]
            sample_batch = batch[REGISTRY_KEYS.BATCH_KEY]

            X = jnp.array(X)
            protein_exp = jnp.array(protein_exp)
            sample_batch = jnp.array(sample_batch)

            _val_loss = evaluate(state, batch_stats, [X, sample_batch, protein_exp])
            n_batches += 1
            val_loss += _val_loss

        losses.append(
            dict(
                metric="stat_train_loss",
                value=loss.item(),
                iterate=epoch,
            )
        )
        losses.append(
            dict(
                metric="stat_val_loss",
                value=val_loss.item() / n_batches,
                iterate=epoch,
            )
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            hasnt_improved = 0
            best_params = copy(state.params)
            best_batch_stats = copy(batch_stats)
        else:
            hasnt_improved += 1
        if hasnt_improved > patience:
            break
    losses = pd.DataFrame(losses)
    return losses, best_params, best_batch_stats


def evaluate_statistic(
    statistic_model,
    params,
    batch_stats,
    val_adata,
    batch_key,
    protein_key,
    batch_size,
    normalize_x=True,
    include_batch_in_input=False,
):
    """Computes the validation loss of the feature importance model."""

    n_samples = val_adata.obs[batch_key].nunique()

    @jax.jit
    def evaluate(inputs):
        x, y = inputs
        x_ = (
            jnp.log1p(1e6 * x / jnp.sum(x, axis=-1, keepdims=True))
            if normalize_x
            else x
        )
        x_ = (
            jnp.concatenate(
                [x_, jax.nn.one_hot(batch_sample.squeeze(-1), n_samples)], axis=-1
            )
            if include_batch_in_input
            else x_
        )
        loss = statistic_model.apply(
            {"params": params, "batch_stats": batch_stats},
            x_,
            y,
            training=False,
        )["loss"]
        return loss

    val_dl = construct_dataloader(
        val_adata,
        batch_size=batch_size,
        shuffle=False,
        batch_key=batch_key,
        protein_key=protein_key,
    )

    val_loss = 0
    n_batches = 0
    for batch in val_dl:
        X = batch[REGISTRY_KEYS.X_KEY]
        protein_exp = batch["response"]
        batch_sample = batch[REGISTRY_KEYS.BATCH_KEY]

        X = jnp.array(X)
        protein_exp = jnp.array(protein_exp)
        batch_sample = jnp.array(batch_sample)

        _val_loss = evaluate([X, batch_sample, protein_exp])
        n_batches += 1
        val_loss += _val_loss
    return val_loss.item() / n_batches
