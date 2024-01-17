from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, logging
from ml_collections import config_flags

from config import load_cfgs
from logistic_regression import (
    plot_density_logistic_regression,
    plot_gradients_logistic_regression_density,
    plot_histograms2d_logistic_regression,
    plot_histograms_logistic_regression,
    plot_logistic_regression_samples,
    Heart,
    German,
    Australian
)

import logistic_regression.statistics as statistics
from sampling.metrics import ess, gelman_rubin_r, effective_sample_size
from sampling import hmc

_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/config.py")


def main(_):

    cfg = load_cfgs(_TASK_FILE)
    cfg.figure_path.mkdir(parents=True, exist_ok=True)

    density = German(batch_size=cfg.sample.num_parallel_chains)
    grad_potential_fn = density.get_grad_energy_fn()

    # average_acceptance_rate = []
    # chains = []

    their_average_eff_sample_size = []
    average_ess_per_second = []

    for i in range(cfg.sample.average_results_over_trials):

        samples, ar, t = hmc(
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
            rng=jax.random.PRNGKey(cfg.seed + i),
        )

        logging.info(f"Acceptance rate: {ar}")

        # for i in range(density.dim):

        #     eff_ess = ess(samples[:, i], density.mean()[i], density.std()[i])
        #     logging.info(f"ESS w_{i}: {eff_ess}")

        # for i in range(density.dim // 4):
        #     plot_logistic_regression_samples(
        #         samples,
        #         num_chains=cfg.sample.num_parallel_chains if cfg.sample.num_parallel_chains > 2 else None,
        #         index=i,
        #         name=cfg.figure_path / Path(f"samples_logistic_regression_{i}.png"),
        #     )
        #     plot_histograms_logistic_regression(
        #         samples,
        #         index=i,
        #         name=cfg.figure_path / Path(f"histograms_logistic_regression_{i}.png"),
        #     )
        #     plot_histograms2d_logistic_regression(
        #         samples,
        #         index=i,
        #         name=cfg.figure_path / Path(f"histograms2d_logistic_regression_{i}.png"),
        #     )
        
        their_eff_ess = effective_sample_size(
                samples[None, :, :density.dim],
                density.mean(),
                density.std()
                )
        their_average_eff_sample_size.append(their_eff_ess)
        for i in range(density.dim):
            logging.info(f"their ESS w_{i}: {their_eff_ess[i]}")
        
        average_ess_per_second.append(their_eff_ess / t)

    their_average_eff_sample_size = np.array(their_average_eff_sample_size)
    their_std_eff_sample_size = np.std(their_average_eff_sample_size, axis=0)
    their_average_eff_sample_size = np.mean(their_average_eff_sample_size, axis=0)

    average_ess_per_second = np.array(average_ess_per_second)
    std_ess_per_second = np.std(average_ess_per_second, axis=0)
    average_ess_per_second = np.mean(average_ess_per_second, axis=0)

    logging.info("--------------")

    for i in range(density.dim):
        logging.info(f"their Average ESS w_{i}: {their_average_eff_sample_size[i]} pm {their_std_eff_sample_size[i]}")
    
    for i in range(density.dim):
        logging.info(f"Average ESS per second w_{i}: {average_ess_per_second[i]} pm {std_ess_per_second[i]}")
    
    # save samples to a file
    if cfg.sample.save_samples:
        cfg.sample.hmc_sample_dir.mkdir(parents=True, exist_ok=True)
        np.save(cfg.sample.hmc_sample_dir / Path(f"hmc_samples_{cfg.dataset.name}.npy"), samples)

if __name__ == "__main__":
    app.run(main)
