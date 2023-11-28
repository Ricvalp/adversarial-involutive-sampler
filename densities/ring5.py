import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import jax


def ring5(x):
    u1 = ((jnp.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) - 1) ** 2) / 0.04
    u2 = ((jnp.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) - 2) ** 2) / 0.04
    u3 = ((jnp.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) - 3) ** 2) / 0.04
    u4 = ((jnp.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) - 4) ** 2) / 0.04
    u5 = ((jnp.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) - 5) ** 2) / 0.04

    return - jnp.min(jnp.array([u1, u2, u3, u4, u5]), axis=0)


def normal(x, mu, cov):
    d = x.shape[0]
    return jnp.exp(-0.5 * jnp.dot(jnp.dot((x - mu).T, jnp.linalg.inv(cov)), (x - mu))) * (1 / jnp.sqrt( 2 * (jnp.pi**d) * jnp.linalg.det(cov)))

normal = jax.vmap(normal, in_axes=(0, None, None))

def hamiltonian_ring5(x, cov_p=jnp.eye(2)*0.5):
    d = x.shape[1]
    return ring5(x[:, : d // 2]) + jnp.log(normal(x[:, d // 2 :], jnp.zeros(2), cov_p))