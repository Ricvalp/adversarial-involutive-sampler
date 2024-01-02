import functools

import jax
import jax.numpy as jnp


def normal(x, mu, inv_cov):
    d = x.shape[0]
    return jnp.exp(-0.5 * jnp.dot(jnp.dot((x - mu).T, inv_cov), (x - mu)))


normal = jax.vmap(normal, in_axes=(0, None, None))


def mog2(x, mu1=jnp.array([5.0, 0.0]), mu2=jnp.array([-5.0, 0.0]), inv_cov=jnp.eye(2) * 2):
    return jnp.log(0.5 * normal(x, mu1, inv_cov) + 0.5 * normal(x, mu2, inv_cov))


def hamiltonian_mog2(
    x,
    mu1=jnp.array([5.0, 0.0]),
    mu2=jnp.array([-5.0, 0.0]),
    inv_cov=jnp.eye(2) * 2,
    inv_cov_p=jnp.eye(2),
):
    d = x.shape[1]
    return -mog2(x[:, : d // 2], mu1, mu2, inv_cov) + jnp.log(
        normal(x[:, d // 2 :], jnp.zeros(2), inv_cov_p)
    )
