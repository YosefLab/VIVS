import flax.linen as nn
import jax.numpy as jnp
import numpyro.distributions as dist
from scvi.module.base import JaxBaseModuleClass, flax_configure, LossOutput
from typing import Dict

from vivs._constants import REGISTRY_KEYS


class JaxBasePredictiveModule(JaxBaseModuleClass):
    """
    Base class for Jax-based predictive models.
    This class simplifies the interface for models that are not
    latent variable models by providing default implementations for
    the `inference`, `generative`, `_get_inference_input`, and
    `_get_generative_input` methods.
    """

    def _get_inference_input(self, tensors: dict[str, jnp.ndarray]):
        """Get input for inference."""
        x = tensors[REGISTRY_KEYS.X_KEY]
        x_log = jnp.log1p(1e6 * x / jnp.sum(x, axis=-1, keepdims=True))
        y = tensors[REGISTRY_KEYS.Y_KEY]
        
        # concat x with batch indices
        batch_indices = tensors[REGISTRY_KEYS.BATCH_KEY].squeeze(-1)
        batch_indices_oh = nn.one_hot(batch_indices, self.n_batch)

        x_concat = jnp.concatenate([x_log, batch_indices_oh], axis=1)

        input_dict = {"X": x_concat, "Y": y}
        return input_dict

    def inference(self, X, Y):
        """Alias for the forward pass of the model."""
        return self.__call__(X, Y)

    def _get_generative_input(self, tensors: Dict, inference_output: Dict):
        """No generative step for predictive models."""
        return {}

    def generative(self, *args, **kwargs):
        """No generative step for predictive models."""
        return {}

    def loss(
        self, tensors, inference_output, generative_output, kl_weight: float = 1.0
    ):
        n_obs_minibatch = tensors[REGISTRY_KEYS.X_KEY].shape[0]
        loss = inference_output["loss"]
        return LossOutput(loss=loss, n_obs_minibatch=n_obs_minibatch)


@flax_configure
class NeuralNet(JaxBasePredictiveModule):
    """
    Importance score relying on a neural network.
    The employed score corresponds to the negative log likelihood
    whose parameters are predicted by a neural network.
    """

    n_hidden: int
    n_features: int
    n_batch: int
    dropout_rate: float
    loss_type: str = "mse"
    training: bool = True

    def setup(self):
        self.dense1 = nn.Dense(features=self.n_hidden)
        self.dense_res = nn.Dense(features=self.n_features)
        self.norm1 = nn.BatchNorm(momentum=0.99, epsilon=0.001)
        self.dropout1 = nn.Dropout(rate=self.dropout_rate)
        self.dense3 = nn.Dense(features=self.n_features)

        self.log_std = 0.0

    def inference(self, X, Y, **kwargs):
        is_eval = not self.training

        h = self.dense1(X)
        h = self.norm1(h, use_running_average=is_eval)
        h = nn.leaky_relu(h)
        h = self.dropout1(h, deterministic=is_eval)

        h = self.dense3(h)

        if self.loss_type == "mse":
            all_loss = -dist.Normal(h, jnp.exp(self.log_std)).log_prob(Y)
        elif self.loss_type == "binary":
            all_loss = -dist.Bernoulli(logits=h).log_prob(Y)
        loss = all_loss.mean()
        return dict(h=h, loss=loss, all_loss=all_loss)


@flax_configure
class LinearModel(JaxBasePredictiveModule):
    """Importance score relying on a linear model."""

    n_features: int
    n_batch: int
    loss_type: str = "mse"
    training: bool = True

    def setup(self):
        self.dense1 = nn.Dense(features=self.n_features)
        self.norm1 = nn.BatchNorm(momentum=0.99, epsilon=0.001)
        self.dropout1 = nn.Dropout(rate=0.0)
        self.log_std = 0.0

    def inference(self, X, Y, **kwargs):
        is_eval = not self.training
        h = self.norm1(X, use_running_average=is_eval)
        h = self.dropout1(h, deterministic=is_eval)
        h = self.dense1(h)
        if self.loss_type == "mse":
            all_loss = -dist.Normal(h, jnp.exp(self.log_std)).log_prob(Y)
        elif self.loss_type == "binary":
            all_loss = -dist.Bernoulli(logits=h).log_prob(Y)
        loss = all_loss.mean()
        return dict(h=h, loss=loss, all_loss=all_loss)
