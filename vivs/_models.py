import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.util import is_prng_key


class ZINB(dist.NegativeBinomial2):
    """Custom ZINB distribution."""

    arg_constraints = {
        "mean": constraints.positive,
        "concentration": constraints.positive,
        "gate": constraints.unit_interval,
    }
    support = constraints.nonnegative_integer

    def __init__(self, mean, concentration, gate, *, validate_args=None):
        self.gate = gate
        super(ZINB, self).__init__(mean, concentration, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        key_bern, key_base = random.split(key)
        shape = sample_shape + self.batch_shape

        samples = super(ZINB, self).sample(key_base, sample_shape=sample_shape)
        mask = random.bernoulli(key_bern, self.gate, shape)
        return jnp.where(mask, 0, samples)

    def log_prob(self, value):
        log_prob = super(ZINB, self).log_prob(value)
        log_prob = jnp.log1p(-self.gate) + log_prob
        return jnp.where(value == 0, jnp.log(self.gate + jnp.exp(log_prob)), log_prob)


class FlaxEncoder(nn.Module):
    """Encoder for Jax VAE."""

    n_input: int
    n_latent: int
    n_hidden: int
    precision: str

    def setup(self):
        """Setup encoder."""
        self.dense1 = nn.Dense(self.n_hidden, precision=self.precision)
        self.dense2 = nn.Dense(self.n_hidden, precision=self.precision)
        self.dense3 = nn.Dense(self.n_latent, precision=self.precision)
        self.dense4 = nn.Dense(self.n_latent, precision=self.precision)

        self.norm1 = nn.BatchNorm(momentum=0.99, epsilon=0.001)
        self.norm2 = nn.BatchNorm(momentum=0.99, epsilon=0.001)

    def __call__(self, x: jnp.ndarray, training: bool = False):
        """Forward pass."""
        is_eval = not training

        x_ = jnp.log1p(x)

        h = self.dense1(x_)
        h = self.norm1(h, use_running_average=is_eval)
        h = nn.relu(h)
        h = self.norm2(h, use_running_average=is_eval)
        h = nn.relu(h)

        mean = self.dense3(h)
        log_var = self.dense4(h)
        return dist.Normal(mean, nn.softplus(log_var))


class FlaxDecoder(nn.Module):
    """Decoder for Jax VAE."""

    n_input: int
    n_hidden: int
    precision: str
    last_h_activation: str = "softmax"

    def setup(self):
        """Setup decoder."""
        self.dense1 = nn.Dense(self.n_hidden, precision=self.precision)
        self.dense2 = nn.Dense(self.n_hidden, precision=self.precision)
        self.dense3 = nn.Dense(self.n_hidden, precision=self.precision)
        self.dense4 = nn.Dense(self.n_hidden, precision=self.precision)
        self.dense5 = nn.Dense(self.n_input, precision=self.precision)

        self.norm1 = nn.BatchNorm(momentum=0.99, epsilon=0.001)
        self.norm2 = nn.BatchNorm(momentum=0.99, epsilon=0.001)
        self.disp = self.param(
            "disp", lambda rng, shape: jax.random.normal(rng, shape), (self.n_input, 1)
        )

        self.zi_logits1 = nn.Dense(self.n_hidden)
        self.zi_logits2 = nn.Dense(self.n_hidden)
        self.zi_logits3 = nn.Dense(self.n_input)
        self.zi_logits_norm = nn.BatchNorm(momentum=0.99, epsilon=0.001)

    def __call__(self, z: jnp.ndarray, batch: jnp.ndarray, training: bool = False):
        """Forward pass."""
        is_eval = not training

        h = self.dense1(z)
        h += self.dense2(batch)

        h = self.norm1(h, use_running_average=is_eval)
        h = nn.relu(h)
        h = self.dense3(h)
        h = self.norm2(h, use_running_average=is_eval)
        h = nn.relu(h)
        h = self.dense5(h)
        if self.last_h_activation == "softmax":
            h = nn.softmax(h)
        elif self.last_h_activation == "softplus":
            h = nn.softplus(h)

        logits = self.zi_logits1(z)
        logits += self.zi_logits2(batch)
        logits = self.zi_logits_norm(logits, use_running_average=is_eval)
        logits = nn.relu(logits)
        logits = self.zi_logits3(logits)
        probs = nn.sigmoid(logits)
        return h, self.disp.ravel(), probs


class JAXSCVAE(nn.Module):
    n_input: int
    n_latent: int
    n_hidden: int
    precision: str = None
    likelihood: str = "nb"
    dropout_rate: float = 0.0
    last_h_activation: str = "softmax"

    def setup(self):
        self.encoder = FlaxEncoder(
            self.n_input, self.n_latent, self.n_hidden, precision=self.precision
        )
        self.decoder = FlaxDecoder(
            self.n_input,
            self.n_hidden,
            precision=self.precision,
            last_h_activation=self.last_h_activation,
        )
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(
        self,
        x,
        batch_indices,
        n_samples=1,
        training: bool = False,
        use_prior=False,
        kl_weight=1.0,
    ):
        z_rng = self.make_rng("z")
        sample_shape = () if n_samples == 1 else (n_samples,)
        x_ = jnp.log1p(x)
        x_ = self.dropout(x_, deterministic=not training)
        if use_prior:
            qz = dist.Normal(0, 1)
        else:
            qz = self.encoder(x_, training=training)
        z = qz.rsample(z_rng, sample_shape=sample_shape)

        h, disp, probs = self.decoder(z, batch_indices, training=training)
        if self.last_h_activation == "softmax":
            library = x.sum(-1, keepdims=True)
            scale = h * library
        else:
            scale = h
        if self.likelihood == "nb":
            px = dist.NegativeBinomial2(scale, jnp.exp(disp))
        elif self.likelihood == "zinb":
            px = ZINB(scale, jnp.exp(disp), probs)
        else:
            px = dist.Poisson(scale)
        log_px = px.log_prob(x).sum(-1)
        kl = dist.kl_divergence(qz, dist.Normal(0, 1)).sum(-1)
        elbo = log_px - (kl_weight * kl)
        loss = -elbo.mean()
        reconstruction_loss = -log_px.mean()
        return dict(
            loss=loss, h=h, z=z, px=px, reconstruction_loss=reconstruction_loss, qz=qz
        )


class ImportanceScorer(nn.Module):
    """
    Importance score relying on a neural network.

    The employed score corresponds to the negative log likelihood
    whose parameters are predicted by a neural network.
    """

    n_hidden: int
    n_features: int
    dropout_rate: float
    loss_type: str = "mse"

    def setup(self):
        self.dense1 = nn.Dense(features=self.n_hidden)
        self.dense_res = nn.Dense(features=self.n_features)
        self.norm1 = nn.BatchNorm(momentum=0.99, epsilon=0.001)
        self.dropout1 = nn.Dropout(rate=self.dropout_rate)
        self.dense3 = nn.Dense(features=self.n_features)

        self.log_std = 0.0

    def __call__(self, x, y, training: bool = False):
        is_eval = not training

        h = self.dense1(x)
        h = self.norm1(h, use_running_average=is_eval)
        h = nn.leaky_relu(h)
        h = self.dropout1(h, deterministic=is_eval)

        h = self.dense3(h)

        if self.loss_type == "mse":
            all_loss = -dist.Normal(h, jnp.exp(self.log_std)).log_prob(y)
        elif self.loss_type == "binary":
            all_loss = -dist.Bernoulli(logits=h).log_prob(y)
        loss = all_loss.mean()
        return dict(h=h, loss=loss, all_loss=all_loss)


class ImportanceScorerLinear(nn.Module):
    """Importance score relying on a linear model."""

    n_features: int
    loss_type: str = "mse"

    def setup(self):
        self.dense1 = nn.Dense(features=self.n_features)
        self.norm1 = nn.BatchNorm(momentum=0.99, epsilon=0.001)
        self.dropout1 = nn.Dropout(rate=0.0)
        self.log_std = 0.0

    def __call__(self, x, y, training: bool = False):
        is_eval = not training
        h = self.norm1(x, use_running_average=is_eval)
        h = self.dropout1(h, deterministic=is_eval)
        h = self.dense1(h)
        if self.loss_type == "mse":
            all_loss = -dist.Normal(h, jnp.exp(self.log_std)).log_prob(y)
        elif self.loss_type == "binary":
            all_loss = -dist.Bernoulli(logits=h).log_prob(y)
        loss = all_loss.mean()
        return dict(h=h, loss=loss, all_loss=all_loss)
