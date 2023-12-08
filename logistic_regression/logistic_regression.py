import jax
import jax.numpy as jnp
from jax import grad


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x) + 1e-6)


def cond_log_likelihood(t, X, w):
    return jnp.sum(t * jnp.log(sigmoid(X @ w)) + (1 - t) * jnp.log(1 - sigmoid(X @ w)))


def log_prior(w, cov):
    d = w.shape[0]
    return (
        -0.5
        * jnp.dot(jnp.dot(w.T, jnp.linalg.inv(cov)), w)
        * jnp.log(1 / jnp.sqrt(2 * (jnp.pi**d) * jnp.linalg.det(cov)))
    )


def log_posterior(w, t, X, cov):
    return cond_log_likelihood(t, X, w) + log_prior(w, cov)


vmap_log_posterior = jax.vmap(log_posterior, in_axes=(0, None, None, None))

grad_log_posterior = jax.vmap(grad(log_posterior), in_axes=(0, None, None, None))


def normalize_covariates(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


def normal(x, mu, cov):
    d = x.shape[0]
    return jnp.exp(-0.5 * jnp.dot(jnp.dot((x - mu).T, jnp.linalg.inv(cov)), (x - mu))) * (
        1 / jnp.sqrt(2 * (jnp.pi**d) * jnp.linalg.det(cov))
    )


normal = jax.vmap(normal, in_axes=(0, None, None))


def hamiltonian_logistic_regression(w, t, X, posterior_cov):
    d = w.shape[1]
    cov_p = jnp.eye(d // 2) * 0.5
    return vmap_log_posterior(w[:, : d // 2], t, X, posterior_cov) + jnp.log(
        normal(w[:, d // 2 :], jnp.zeros(d // 2), cov_p)
    )


# X_data = jax.random.normal(jax.random.PRNGKey(1), (100, 14))
# X = jnp.concatenate([X_data, jnp.ones((X_data.shape[0], 1))], axis=1)
# w = jax.random.normal(jax.random.PRNGKey(0), (50, 30))
# t = jax.random.bernoulli(jax.random.PRNGKey(1), p=0.5, shape=(100, 1))

# hamiltonian_logistic_regression(w, t, X, posterior_cov=jnp.eye(15))
