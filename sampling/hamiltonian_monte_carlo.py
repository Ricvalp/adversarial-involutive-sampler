import time

import jax
import jax.numpy as jnp
from absl import logging
from jax import random


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
    log_prob_ratio = log_prob_new - log_prob_old

    accept = jnp.log(jax.random.uniform(accept_subkey, (parallel_chains,))) < log_prob_ratio

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

    return jnp.vstack(samples), jnp.array(ars).mean()


# # Example usage
# import matplotlib.pyplot as plt
# import time

# num_samples = 100
# num_steps = 40
# step_size = 0.01

# coordinates = jax.random.uniform(random.PRNGKey(42), (10000, 2), minval=-5, maxval=5)

# time_start = time.time()
# vectors = grad_mog6(coordinates)
# print(f"Time taken to compile: {time.time() - time_start}")

# time_start = time.time()
# vectors = grad_mog6(coordinates)
# print(f"Time taken to get the grads: {time.time() - time_start}")

# plt.figure(figsize=(8, 6))

# # Extracting x and y components of vectors
# U = vectors[:, 0]
# V = vectors[:, 1]

# # Plotting the vectors
# plt.quiver(
#     coordinates[:, 0],
#     coordinates[:, 1],
#     U,
#     V,
#     scale=100,
#     width=0.001,
#     headwidth=0.1,
# )

# # Set plot title and labels
# plt.title('2D Gradient Field')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.savefig("figures/hamiltonian_monte_carlo_gradients.png")

# time_start = time.time()
# samples = hmc(
#     hmc_proposal,
#     hamiltonian_mog6,
#     grad_mog6,
#     2,
#     num_samples,
#     jnp.eye(2)*0.5,
#     parallel_chains=1000,
#     burn_in=1000,
#     rng=jax.random.PRNGKey(42)
# )
# print(f"Time taken to sample: {time.time() - time_start}")

# plt.scatter(samples[0][:, 0], samples[0][:, 1], alpha=0.5, s=0.1)
# plt.savefig("figures/hamiltonian_monte_carlo_samples.png")
