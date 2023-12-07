import jax
import jax.numpy as jnp
from jax import grad


def normal(x, mu, cov):
    d = x.shape[0]
    return jnp.exp(-0.5 * jnp.dot(jnp.dot((x - mu).T, jnp.linalg.inv(cov)), (x - mu))) * (
        1 / jnp.sqrt(2 * (jnp.pi**d) * jnp.linalg.det(cov))
    )


normal = jax.vmap(normal, in_axes=(0, None, None))


def mog6(x):
    mus = [5 * jnp.array([jnp.sin(jnp.pi * i / 3), jnp.cos(jnp.pi * i / 3)]) for i in range(6)]
    cov = jnp.eye(2) * 0.5
    return jnp.log(jnp.sum(jnp.array([normal(x, mu, cov) for mu in mus]), axis=0) / 6)


def hamiltonian_mog6(x, cov_p=jnp.eye(2) * 0.5):
    d = x.shape[1]
    return mog6(x[:, : d // 2]) + jnp.log(normal(x[:, d // 2 :], jnp.zeros(2), cov_p))


def nv_normal(x, mu, cov, d):
    return jnp.exp(-0.5 * jnp.dot(jnp.dot((x - mu).T, jnp.linalg.inv(cov)), (x - mu))) * (
        1 / jnp.sqrt(2 * (jnp.pi**d) * jnp.linalg.det(cov))
    )


def nv_mog6(x):
    d = x.shape[0]
    mus = [5 * jnp.array([jnp.sin(jnp.pi * i / 3), jnp.cos(jnp.pi * i / 3)]) for i in range(6)]
    cov = jnp.eye(2) * 0.5
    return jnp.log(jnp.sum(jnp.array([nv_normal(x, mu, cov, d) for mu in mus]), axis=0) / 6)


grad_mog6 = jax.jit(jax.vmap(grad(nv_mog6)))
