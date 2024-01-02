import jax
import jax.numpy as jnp
from jax import grad


def normal(x, mu, inv_cov):
    return jnp.exp(-0.5 * jnp.dot(jnp.dot((x - mu).T, inv_cov), (x - mu)))


normal = jax.vmap(normal, in_axes=(0, None, None))


def mog6(x):
    mus = [5 * jnp.array([jnp.sin(jnp.pi * i / 3), jnp.cos(jnp.pi * i / 3)]) for i in range(6)]
    inv_cov = jnp.eye(2) * 2
    return jnp.log(jnp.sum(jnp.array([normal(x, mu, inv_cov) for mu in mus]), axis=0) / 6)


def hamiltonian_mog6(x, inv_cov_p=jnp.eye(2) * 2):
    d = x.shape[1]
    return -mog6(x[:, : d // 2]) - jnp.log(normal(x[:, d // 2 :], jnp.zeros(2), inv_cov_p))


def nv_normal(x, mu, inv_cov):
    return jnp.exp(-0.5 * jnp.dot(jnp.dot((x - mu).T, inv_cov), (x - mu)))


def nv_mog6(x):
    mus = [5 * jnp.array([jnp.sin(jnp.pi * i / 3), jnp.cos(jnp.pi * i / 3)]) for i in range(6)]
    inv_cov = jnp.eye(2) * 2
    return -jnp.log(jnp.sum(jnp.array([nv_normal(x, mu, inv_cov) for mu in mus]), axis=0) / 6)


grad_mog6 = jax.vmap(grad(nv_mog6))
