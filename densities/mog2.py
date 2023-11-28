import jax
import jax.numpy as jnp
import functools

def normal(x, mu, cov):
    d = x.shape[0]
    return jnp.exp(-0.5 * jnp.dot(jnp.dot((x - mu).T, jnp.linalg.inv(cov)), (x - mu))) * (1 / jnp.sqrt( 2 * (jnp.pi**d) * jnp.linalg.det(cov)))

normal = jax.vmap(normal, in_axes=(0, None, None))

def mog2(x, mu1=jnp.array([5., 0.]), mu2=jnp.array([-5., 0.]), cov=jnp.eye(2)*0.5):
    return jnp.log(0.5 * normal(x, mu1, cov) + 0.5 * normal(x, mu2, cov))

def hamiltonian_mog2(x, mu1=jnp.array([5., 0.]), mu2=jnp.array([-5., 0.]), cov=jnp.eye(2)*0.5, cov_p=jnp.eye(2)*0.5):
    d = x.shape[1]
    return mog2(x[:, : d // 2], mu1, mu2, cov) + jnp.log(normal(x[:, d // 2 :], jnp.zeros(2), cov_p))
