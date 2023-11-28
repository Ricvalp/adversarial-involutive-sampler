import jax
import jax.numpy as jnp


def mh_kernel_with_momentum(x, key,  cov_p, kernel, density, parallel_chains=100):
        
        key, accept_subkey, momentum_subkey = jax.random.split(key, 3)
        x_new = kernel(x)

        log_prob_new = density(x_new)
        log_prob_old = density(x)
        log_prob_ratio = log_prob_new - log_prob_old

        accept = jnp.log(jax.random.uniform(accept_subkey, (parallel_chains,))) < log_prob_ratio

        x_new = jnp.where(accept[:, None], x_new, x)[:, : x.shape[1] // 2]
        momentum = jax.random.multivariate_normal(momentum_subkey, jnp.zeros(x.shape[1] // 2), cov_p, (parallel_chains,))        
        x_new = jnp.concatenate([x_new, momentum], axis=1)

        return x_new, key

jit_mh_kernel_with_momentum = jax.jit(mh_kernel_with_momentum, static_argnums=(3, 4, 5, ))

def metropolis_hastings_with_momentum(kernel, density, d,  n, cov_p, parallel_chains=100, burn_in=100, rng=jax.random.PRNGKey(42)):

    first_init_subkey, second_init_subkey, sampling_subkey = jax.random.split(rng, 3)

    x = jax.random.normal(first_init_subkey, (parallel_chains, d))
    x = jnp.concatenate([x, jax.random.multivariate_normal(second_init_subkey, jnp.zeros(d), cov_p, (parallel_chains,))], axis=1)

    samples = []
    for i in range(n + burn_in):
        x, sampling_subkey = jit_mh_kernel_with_momentum(x, sampling_subkey, cov_p, kernel, density, parallel_chains=parallel_chains)
        if i >= burn_in:
            samples.append(x)

    return jnp.vstack(samples)