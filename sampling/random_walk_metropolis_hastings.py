import jax
import jax.numpy as jnp

def mh_kernel(x, key, kernel, density, parallel_chains=100):

        key, kernel_subkey, accept_subkey = jax.random.split(key, 3)
        x_new, key = kernel(x, kernel_subkey)
    
        log_prob_new = density(x_new)
        log_prob_old = density(x)
        log_prob_ratio = log_prob_old - log_prob_new

        accept = jnp.log(jax.random.uniform(accept_subkey, (parallel_chains,))) < log_prob_ratio

        return jnp.where(accept[:, None], x_new, x), key

jit_mh_kernel = jax.jit(mh_kernel, static_argnums=(2, 3, 4, ))

def metropolis_hastings(kernel, density, d, n, parallel_chains=100, burn_in=100, seed=42):

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)

    x = jax.random.normal(subkey, (parallel_chains, d))

    samples = []
    for i in range(n + burn_in):
        x, key = jit_mh_kernel(x, key, kernel, density, parallel_chains=parallel_chains)
        if i >= burn_in:
            samples.append(x)

    return jnp.vstack(samples)
