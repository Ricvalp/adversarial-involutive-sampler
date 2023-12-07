import jax
import jax.numpy as jnp
from jax import grad


def ring(x):
    return -((jnp.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) - 2) ** 2) / 0.32


def normal(x, mu, cov):
    d = x.shape[0]
    return jnp.exp(-0.5 * jnp.dot(jnp.dot((x - mu).T, jnp.linalg.inv(cov)), (x - mu))) * (
        1 / jnp.sqrt(2 * (jnp.pi**d) * jnp.linalg.det(cov))
    )


normal = jax.vmap(normal, in_axes=(0, None, None))


def hamiltonian_ring(x, cov_p=jnp.eye(2) * 0.5):
    d = x.shape[1]
    return ring(x[:, : d // 2]) + jnp.log(normal(x[:, d // 2 :], jnp.zeros(2), cov_p))


def nv_ring(x):
    return -((jnp.sqrt(x[0] ** 2 + x[1] ** 2) - 2) ** 2) / 0.32


grad_ring = jax.jit(jax.vmap(grad(nv_ring)))
