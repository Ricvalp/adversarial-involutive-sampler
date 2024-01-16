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


def gelman_rubin_r(chains):

    m = chains.shape[0]
    n = chains.shape[1]
    psi = np.sqrt(np.mean(chains ** 2, axis=-1))
    psi_bar = np.mean(psi)
    psi_j_bar = np.mean(psi, axis=1)
    B = (n / (m - 1)) * np.sum((psi_j_bar - psi_bar) ** 2)
    W = (1 / (m * (n-1))) * np.sum(
        np.sum((psi - psi_j_bar[:, None]) ** 2, axis=-1)
    )
    sigma_hat_squared = (((n - 1) / n) * W) + (B / n)
    V = sigma_hat_squared + (B / (n*m))
    R_hat = V / W
    # R_hat = ((m+1)/m) * (sigma_hat_squared / W) - ((n-1) / (m * n))
    return R_hat


x = np.random.normal(0, 1, (5, 100, 2))
y = np.random.normal(1, 1, (5, 100, 2))
x = np.concatenate([x, y], axis=0)
gelman_rubin_r(x)