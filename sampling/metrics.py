import jax
import jax.numpy as jnp
import numpy as np


def lag_s_autocorrelation(samples, s, mu, sigma):
    N = samples.shape[0]
    c = 1 / ((sigma**2) * (N - s))
    summand = (samples[s:] - mu) * (samples[:-s] - mu)
    return c * np.sum(summand)


def ess(samples, mu, sigma):
    samples = np.array(samples)
    N = samples.shape[0]

    summ = 0
    for s in range(1, N):
        rho = lag_s_autocorrelation(samples, s, mu, sigma)
        if rho < 0.05:
            break
        summ += (1 - s / N) * rho

    return N / (1 + 2 * summ)
