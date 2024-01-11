import time

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging


def leapfrog_step(x, epsilon, grad_U):
    q = x[:, : x.shape[1] // 2]
    p = x[:, x.shape[1] // 2 :]
    p = p - 0.5 * epsilon * grad_U(q)
    q = q + epsilon * p
    p = p - 0.5 * epsilon * grad_U(q)
    return jnp.concatenate([q, p], axis=1)


def hmc_proposal(x, grad_potential_fn, num_steps, step_size):
    # Leapfrog integration
    for _ in range(num_steps):
        x = leapfrog_step(x, step_size, grad_potential_fn)

    return x


def hmc_kernel(
    x, key, cov_p, density, num_steps, step_size, grad_potential_fn, parallel_chains=100
):
    key, accept_subkey, momentum_subkey = jax.random.split(key, 3)
    x_new = hmc_proposal(x, grad_potential_fn, num_steps, step_size)

    log_prob_new = density(x_new)
    log_prob_old = density(x)
    log_prob_ratio = log_prob_new - log_prob_old  # log_prob_new - log_prob_old

    accept = jax.random.uniform(accept_subkey, (parallel_chains,)) < jnp.exp(log_prob_ratio)

    x_new = jnp.where(accept[:, None], x_new, x)[:, : x.shape[1] // 2]
    momentum = jax.random.multivariate_normal(
        momentum_subkey, jnp.zeros(x.shape[1] // 2), cov_p, (parallel_chains,)
    )
    x_new = jnp.concatenate([x_new, momentum], axis=1)

    return x_new, accept.mean(), key


jit_hmc_kernel = jax.jit(
    hmc_kernel,
    static_argnums=(
        3,
        4,
        5,
        6,
        7,
    ),
)



def hmc(
    density,
    grad_potential_fn,
    d,
    n,
    cov_p,
    num_steps,
    step_size,
    parallel_chains=100,
    burn_in=100,
    initial_std=1.,
    rng=jax.random.PRNGKey(42),
):
    first_init_subkey, second_init_subkey, sampling_subkey = jax.random.split(rng, 3)

    x = jax.random.normal(first_init_subkey, (parallel_chains, d))*initial_std
    x = jnp.concatenate(
        [
            x,
            jax.random.multivariate_normal(
                second_init_subkey, jnp.zeros(d), cov_p, (parallel_chains,)
            ),
        ],
        axis=1,
    )

    logging.info("Jitting HMC kernel...")
    t = time.time()
    jit_hmc_kernel(
        x,
        sampling_subkey,
        cov_p,
        density,
        num_steps,
        step_size,
        grad_potential_fn,
        parallel_chains=parallel_chains,
    )
    logging.info(f"Jitting done. Time taken: {time.time() - t}")

    samples = []
    ars = []
    logging.info("Sampling...")
    time_start = time.time()
    for i in range(n + burn_in):
        x, ar, sampling_subkey = jit_hmc_kernel(
            x,
            sampling_subkey,
            cov_p,
            density,
            num_steps,
            step_size,
            grad_potential_fn,
            parallel_chains=parallel_chains,
        )
        if i >= burn_in:
            samples.append(x)
            ars.append(ar)
    logging.info(f"Sampling done. Time taken: {time.time() - time_start}")

    return np.vstack(samples), np.array(ars).mean()

# returns also hamiltonian trajectories

def hmc_proposal_debug(x, grad_potential_fn, num_steps, step_size):
    # Leapfrog integration
    tr = [x]
    for _ in range(num_steps):
        x = leapfrog_step(x, step_size, grad_potential_fn)
        tr.append(x)

    return x, jnp.array(tr)

def hmc_kernel_debug(
    x, key, cov_p, density, num_steps, step_size, grad_potential_fn, parallel_chains=100
):
    key, accept_subkey, momentum_subkey = jax.random.split(key, 3)
    x_new, tr = hmc_proposal_debug(x, grad_potential_fn, num_steps, step_size)

    log_prob_new = density(x_new)
    log_prob_old = density(x)
    log_prob_ratio = log_prob_old - log_prob_new

    accept = jnp.log(jax.random.uniform(accept_subkey, (parallel_chains,))) < log_prob_ratio

    x_new = jnp.where(accept[:, None], x_new, x)[:, : x.shape[1] // 2]
    momentum = jax.random.multivariate_normal(
        momentum_subkey, jnp.zeros(x.shape[1] // 2), cov_p, (parallel_chains,)
    )
    x_new = jnp.concatenate([x_new, momentum], axis=1)

    return x_new, accept.mean(), key, tr


jit_hmc_kernel_debug = jax.jit(
    hmc_kernel_debug,
    static_argnums=(
        3,
        4,
        5,
        6,
        7,
    ),
)

def hmc_debug(
    density,
    grad_potential_fn,
    d,
    n,
    cov_p,
    num_steps,
    step_size,
    parallel_chains=100,
    burn_in=100,
    rng=jax.random.PRNGKey(42),
):
    first_init_subkey, second_init_subkey, sampling_subkey = jax.random.split(rng, 3)

    x = jax.random.normal(first_init_subkey, (parallel_chains, d))
    x = jnp.concatenate(
        [
            x,
            jax.random.multivariate_normal(
                second_init_subkey, jnp.zeros(d), cov_p, (parallel_chains,)
            ),
        ],
        axis=1,
    )

    logging.info("Jitting HMC kernel...")
    t = time.time()
    jit_hmc_kernel_debug(
        x,
        sampling_subkey,
        cov_p,
        density,
        num_steps,
        step_size,
        grad_potential_fn,
        parallel_chains=parallel_chains,
    )
    logging.info(f"Jitting done. Time taken: {time.time() - t}")

    samples = []
    ars = []
    trs = []
    logging.info("Sampling...")
    time_start = time.time()
    for i in range(n + burn_in):
        x, ar, sampling_subkey, tr = jit_hmc_kernel_debug(
            x,
            sampling_subkey,
            cov_p,
            density,
            num_steps,
            step_size,
            grad_potential_fn,
            parallel_chains=parallel_chains,
        )
        if i >= burn_in:
            samples.append(x)
            ars.append(ar)
            trs.append(tr)
    logging.info(f"Sampling done. Time taken: {time.time() - time_start}")

    return np.vstack(samples), np.array(ars).mean(), jnp.array(trs)
