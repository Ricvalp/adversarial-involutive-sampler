import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad


statistics_ring5 = {
    'mu': [0., 0.],
    'sigma': [2.58, 2.58]
}

def ring5(x):
    u1 = ((jnp.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) - 1) ** 2) / 0.04
    u2 = ((jnp.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) - 2) ** 2) / 0.04
    u3 = ((jnp.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) - 3) ** 2) / 0.04
    u4 = ((jnp.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) - 4) ** 2) / 0.04
    u5 = ((jnp.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) - 5) ** 2) / 0.04

    return jnp.min(jnp.array([u1, u2, u3, u4, u5]), axis=0)


def log_normal(x, mu, inv_cov):
    return -0.5 * jnp.dot(jnp.dot((x - mu).T, inv_cov), (x - mu))


log_normal = jax.vmap(log_normal, in_axes=(0, None, None))


def hamiltonian_ring5(x, inv_cov_p=jnp.eye(2)):
    d = x.shape[1]
    return ring5(x[:, : d // 2]) - log_normal(x[:, d // 2 :], jnp.zeros(2), inv_cov_p)


def nv_ring5(x):
    u1 = ((jnp.sqrt(x[0] ** 2 + x[1] ** 2) - 1) ** 2) / 0.04
    u2 = ((jnp.sqrt(x[0] ** 2 + x[1] ** 2) - 2) ** 2) / 0.04
    u3 = ((jnp.sqrt(x[0] ** 2 + x[1] ** 2) - 3) ** 2) / 0.04
    u4 = ((jnp.sqrt(x[0] ** 2 + x[1] ** 2) - 4) ** 2) / 0.04
    u5 = ((jnp.sqrt(x[0] ** 2 + x[1] ** 2) - 5) ** 2) / 0.04

    return jnp.min(jnp.array([u1, u2, u3, u4, u5]), axis=0)


grad_ring5 = jax.vmap(grad(nv_ring5))


# import matplotlib.pyplot as plt

# x = jnp.linspace(-8, 8, 100)
# X, Y = jnp.meshgrid(x, x)
# Z = jnp.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])
# grads = grad_ring5(Z)
# density = jnp.exp(-ring5(Z)).reshape((100, 100))
# plt.imshow(density)
# plt.colorbar()
# # plt.quiver(Z[:, 0], Z[:, 1], grads[:, 0], grads[:, 1])
# plt.show()