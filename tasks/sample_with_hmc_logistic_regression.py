from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pymc3 as pm
from absl import app, logging
from ml_collections import config_flags

# from sampling.metrics import ess
from sklearn.datasets import fetch_openml

from config import load_cfgs
from logistic_regression import (
    grad_log_posterior,
    hamiltonian_logistic_regression,
    normalize_covariates,
    plot_logistic_regression_samples,
)
from sampling import hmc

_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/config.py")


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    cfg.figure_path.mkdir(parents=True, exist_ok=True)

    heart_data = fetch_openml(name="heart", version=1)
    X_data = heart_data.data.toarray()
    X_data = normalize_covariates(X_data)
    X = jnp.concatenate([X_data, jnp.ones((X_data.shape[0], 1))], axis=1)
    t = heart_data.target

    density = lambda x: hamiltonian_logistic_regression(
        x, t, X, cov=jnp.eye(X.shape[1]) * 100, cov_p=jnp.eye(X.shape[1])
    )
    grad_potential_fn = lambda x: grad_log_posterior(x, t, X, jnp.eye(X.shape[1]) * 100)

    samples, ar = hmc(
        density=density,
        grad_potential_fn=grad_potential_fn,
        cov_p=jnp.eye(X.shape[1]),
        d=X.shape[1],
        parallel_chains=cfg.sample.num_parallel_chains,
        num_steps=cfg.hmc.num_steps,
        step_size=cfg.hmc.step_size,
        n=cfg.sample.num_iterations,
        burn_in=cfg.sample.burn_in,
        rng=jax.random.PRNGKey(cfg.seed),
    )

    logging.info(f"Acceptance rate: {ar}")
    for i in range(X.shape[1]):
        logging.info(f"ESS for w_{i}: {pm.ess(np.array(samples[:, i]))}")

    for i in range(X.shape[1] - 1):
        plot_logistic_regression_samples(
            samples,
            i=i,
            j=i + 1,
            name=cfg.figure_path / Path(f"samples_logistic_regression_{i}.png"),
        )


if __name__ == "__main__":
    app.run(main)
