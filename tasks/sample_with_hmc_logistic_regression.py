from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pymc3 as pm
from absl import app, logging
from ml_collections import config_flags
from sklearn.datasets import fetch_openml

from config import load_cfgs
from logistic_regression import (
    generate_dataset,
    get_predictions,
    grad_U,
    hamiltonian,
    normalize_covariates,
    plot_density_logistic_regression,
    plot_gradients_logistic_regression_density,
    plot_histograms2d_logistic_regression,
    plot_histograms_logistic_regression,
    plot_logistic_regression_samples,
)
from sampling import hmc

_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/config.py")


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    cfg.figure_path.mkdir(parents=True, exist_ok=True)

    heart_data = fetch_openml(name="heart", version=1)
    X_data = heart_data.data.toarray()
    # X_data = normalize_covariates(X_data)
    X = X_data[:, : cfg.dataset.num_covariates]
    X = jnp.concatenate([X, jnp.ones((X.shape[0], 1))], axis=1)
    t_data = heart_data.target
    t = t_data  # [:cfg.dataset.test_split]
    t = (t == 1).astype(int).astype(float)

    # w=jnp.array([-2.5, 1.5, .5])
    # t, X = generate_dataset(
    #     n=25,
    #     w=w,
    #     rng=jax.random.PRNGKey(1)
    # )

    density = lambda x: hamiltonian(x, t, X, inv_sigma=jnp.eye(X.shape[1]) * 0.01)
    grad_potential_fn = lambda x: grad_U(x, t, X, jnp.eye(X.shape[1]))

    # plot_density_logistic_regression(l=6., w=w, density=density, name="density.png")
    # plot_gradients_logistic_regression_density(l=6., w=w, grad_potential_fn=grad_potential_fn, name="gradients.png")

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
            num_chains=None,  # cfg.sample.num_parallel_chains,
            name=cfg.figure_path / Path(f"samples_logistic_regression_{i}.png"),
        )
        plot_histograms_logistic_regression(
            samples,
            i=i,
            name=cfg.figure_path / Path(f"histograms_logistic_regression_{i}.png"),
        )
        plot_histograms2d_logistic_regression(
            samples,
            i=i,
            j=i + 1,
            name=cfg.figure_path / Path(f"histograms2d_logistic_regression_{i}.png"),
        )

    predictions = get_predictions(X, samples[:, : X.shape[1]])
    logging.info(f"Accuracy: {np.mean(predictions == t.astype(int))}")

    assert True


if __name__ == "__main__":
    app.run(main)
