import jax
import jax.numpy as jnp


def random_walk_kernel(sigma=1.):
        
        def kernel(x, key):
            key, subkey = jax.random.split(key, 2)
            return x + jax.random.normal(subkey, x.shape) * sigma, key
    
        return jax.jit(kernel)