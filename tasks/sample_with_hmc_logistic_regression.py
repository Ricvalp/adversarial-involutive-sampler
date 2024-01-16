from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
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
    Heart
)

import logistic_regression.statistics as statistics
from sampling.metrics import ess, gelman_rubin_r
from sampling import hmc

_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/config.py")


def main(_):

    cfg = load_cfgs(_TASK_FILE)
    cfg.figure_path.mkdir(parents=True, exist_ok=True)

    density = Heart(batch_size=cfg.sample.num_parallel_chains)
    grad_potential_fn = density.get_grad_energy_fn()

    # plot_density_logistic_regression(lim=6., w=None, d=density.dim, density=density, name="density.png")
    # plot_gradients_logistic_regression_density(lim=6., w=None, d=density.dim, grad_potential_fn=grad_potential_fn, name="gradients.png")

    samples, ar = hmc(
        density=density,
        grad_potential_fn=grad_potential_fn,
        cov_p=jnp.eye(density.dim),
        d=density.dim,
        parallel_chains=cfg.sample.num_parallel_chains,
        num_steps=cfg.hmc.num_steps,
        step_size=cfg.hmc.step_size,
        n=cfg.sample.num_iterations,
        burn_in=cfg.sample.burn_in,
        initial_std=.1,
        rng=jax.random.PRNGKey(cfg.seed),
    )

    logging.info(f"Acceptance rate: {ar}")
    for i in range(density.dim):

        eff_ess = ess(samples[:, i], density.mean()[i], density.std()[i])
        logging.info(f"ESS w_{i}: {eff_ess}")

    for i in range(density.dim // 4):
        plot_logistic_regression_samples(
            samples,
            num_chains=cfg.sample.num_parallel_chains if cfg.sample.num_parallel_chains > 2 else None,
            index=i,
            name=cfg.figure_path / Path(f"samples_logistic_regression_{i}.png"),
        )
        plot_histograms_logistic_regression(
            samples,
            index=i,
            name=cfg.figure_path / Path(f"histograms_logistic_regression_{i}.png"),
        )
        plot_histograms2d_logistic_regression(
            samples,
            index=i,
            name=cfg.figure_path / Path(f"histograms2d_logistic_regression_{i}.png"),
        )


    # data = np.load(f'data/{cfg.dataset.name}/data.npy')
    # labels = np.load(f'data/{cfg.dataset.name}/labels.npy')
    # X_data = normalize_covariates(data)
    # X = X_data[:, : cfg.dataset.num_covariates]
    # X = jnp.concatenate([X, jnp.ones((X.shape[0], 1))], axis=1)
    # t = labels[:, 0]
    # t = (t == 1).astype(int).astype(float)

    # X = jnp.concatenate([density.data[0], jnp.ones((density.data[0].shape[0], 1))], axis=1)
    # predictions = get_predictions(X, samples[:, : density.dim])
    # logging.info(f"Accuracy: {np.mean(predictions == density.labels[:, 0].astype(int))}")


    # save samples to a file
    if cfg.sample.save_samples:
        cfg.sample.hmc_sample_dir.mkdir(parents=True, exist_ok=True)
        np.save(cfg.sample.hmc_sample_dir / Path(f"hmc_samples_{cfg.dataset.name}.npy"), samples)

    assert True


if __name__ == "__main__":
    app.run(main)
