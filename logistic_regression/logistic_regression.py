import jax
import jax.numpy as jnp
from jax import grad


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def phi(x):
    return


def cond_log_likelihood(t, X, w):
    return jnp.sum(t * jnp.log(sigmoid(X @ w)) + (1 - t) * jnp.log(1 - sigmoid(X @ w)))


def log_prior(w):
    return -w @ w / 2


def log_posterior(w, t, X):
    return cond_log_likelihood(t, X, w) + log_prior(w)


grad_log_posterior = jax.vmap(grad(log_posterior), in_axes=(0, None, None))

log_posterior = jax.jit(jax.vmap(log_posterior, in_axes=(0, None, None)))


def normalize_covariates(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


def normal(x, mu, cov):
    d = x.shape[0]
    return jnp.exp(-0.5 * jnp.dot(jnp.dot((x - mu).T, jnp.linalg.inv(cov)), (x - mu))) * (
        1 / jnp.sqrt(2 * (jnp.pi**d) * jnp.linalg.det(cov))
    )


normal = jax.vmap(normal, in_axes=(0, None, None))


def hamiltonian_logistic_regression(w, t, X):
    d = w.shape[1]
    cov_p = jnp.eye(d // 2) * 0.5
    return log_posterior(w[:, : d // 2], t, X) + jnp.log(
        normal(w[:, d // 2 :], jnp.zeros(d // 2), cov_p)
    )
