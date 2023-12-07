from pathlib import Path

import jax
import jax.numpy as jnp
from absl import app
from ml_collections import config_flags
from sklearn.datasets import fetch_openml

from config import load_cfgs
from densities import plot_logistic_regression_density
from logistic_regression import (
    grad_log_posterior,
    hamiltonian_logistic_regression,
    normalize_covariates,
)
from sampling import hmc, plot_samples, plot_samples_with_momentum

_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/config.py")


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    cfg.figure_path.mkdir(parents=True, exist_ok=True)

    hear_dataset = fetch_openml(name="heart", as_frame="auto")
    X = normalize_covariates(hear_dataset["data"].toarray())
    t = hear_dataset["target"]

    density = lambda x: hamiltonian_logistic_regression(x, t, X)
    grad_potential_fn = lambda x: grad_log_posterior(x, t, X)

    plot_logistic_regression_density(
        density,
        xlim_q=1,
        ylim_q=1,
        xlim_p=1,
        ylim_p=1,
        n=100,
        d=X.shape[1] * 2,
        name=cfg.figure_path / Path("density_logistic_regression.png"),
    )

    samples, ar = hmc(
        density=density,
        grad_potential_fn=grad_potential_fn,
        cov_p=jnp.eye(cfg.sample.d),
        d=cfg.sample.d,
        parallel_chains=cfg.sample.num_parallel_chains,
        num_steps=cfg.hmc.num_steps,
        step_size=cfg.hmc.step_size,
        n=cfg.sample.num_iterations,
        burn_in=cfg.sample.burn_in,
        rng=jax.random.PRNGKey(cfg.seed),
    )

    plot_samples(
        samples,
        target_density=density,
        d=cfg.sample.d * 2,
        ar=ar,
        name=cfg.figure_path / Path("samples_logistic_regression.png"),
    )


if __name__ == "__main__":
    app.run(main)
