import jax
import jax.numpy as jnp
from jax import grad


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x) + 1e-7)


def cond_log_likelihood(t, X, w):
    return jnp.sum(t * jnp.log(sigmoid(X @ w)) + (1 - t) * jnp.log(1 - sigmoid(X @ w)))


def log_prior(w, cov):
    d = w.shape[0]
    return -0.5 * w.T @ jnp.linalg.inv(cov) @ w + jnp.log(
        1 / jnp.sqrt(((2 * jnp.pi) ** d) * jnp.linalg.det(cov))
    )


def log_posterior(w, t, X, cov):
    return cond_log_likelihood(t, X, w) + log_prior(w, cov)


def normalize_covariates(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


vmap_log_posterior = jax.vmap(log_posterior, in_axes=(0, None, None, None))
grad_log_posterior = jax.vmap(grad(log_posterior), in_axes=(0, None, None, None))


def K(w, cov):
    return -0.5 * w @ jnp.linalg.inv(cov) @ w.T


vmap_K = jax.vmap(K, in_axes=(0, None))


def hamiltonian_logistic_regression(w, t, X, cov, cov_p):
    d = w.shape[1] // 2
    return -vmap_log_posterior(w[:, :d], t, X, cov) + vmap_K(w[:, d:], cov_p)


# Fetch the heart dataset
# from sklearn.datasets import fetch_openml
# heart_data = fetch_openml(name='heart', version=1)
# X_data = heart_data.data.toarray()
# X_data = normalize_covariates(X_data)
# X = jnp.concatenate([X_data, jnp.ones((X_data.shape[0], 1))], axis=1)
# t = heart_data.target
# w = jax.random.normal(jax.random.PRNGKey(0), (10, 2 * X.shape[1]))

# assert True
