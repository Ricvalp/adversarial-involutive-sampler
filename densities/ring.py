import jax
import jax.numpy as jnp
from jax import grad


statistics_ring = {
    'mu': [0., 0.],
    'sigma': [1.5, 1.5]
}

def ring(x):
    return ((jnp.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) - 2) ** 2) / 0.32


def log_normal(x, mu, inv_cov):
    d = x.shape[0]
    return -0.5 * jnp.dot(jnp.dot((x - mu).T, inv_cov), (x - mu))


log_normal = jax.vmap(log_normal, in_axes=(0, None, None))


def hamiltonian_ring(x, inv_cov_p=jnp.eye(2)):
    d = x.shape[1]
    return ring(x[:, : d // 2]) - log_normal(x[:, d // 2 :], jnp.zeros(2), inv_cov_p)


def nv_ring(x):
    return ((jnp.sqrt(x[0] ** 2 + x[1] ** 2) - 2) ** 2) / 0.32


grad_ring = jax.vmap(grad(nv_ring))


# import matplotlib.pyplot as plt

# x = jnp.linspace(-8, 8, 100)
# X, Y = jnp.meshgrid(x, x)
# Z = jnp.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])
# grads = grad_ring(Z)
# density = jnp.exp(-ring(Z)).reshape((100, 100))
# plt.imshow(density)
# plt.colorbar()
# # plt.quiver(Z[:, 0], Z[:, 1], grads[:, 0], grads[:, 1])
# plt.show()