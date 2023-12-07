from pathlib import Path

import jax
import jax.numpy as jnp
from absl import app, logging
from ml_collections import config_flags

import densities
from config import load_cfgs
from densities import plot_hamiltonian_density
from sampling import hmc, plot_samples_with_momentum
from sampling.metrics import ess

_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/config.py")


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    cfg.figure_path.mkdir(parents=True, exist_ok=True)

    density = getattr(densities, cfg.target_density.name)
    grad_potential_fn = getattr(densities, f"grad_{cfg.hmc.potential_function_name}")

    plot_hamiltonian_density(
        density,
        xlim_q=6,
        ylim_q=6,
        xlim_p=6,
        ylim_p=6,
        n=100,
        q_0=0.0,
        q_1=0.0,
        name=cfg.figure_path / Path("hamiltonian_density.png"),
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

    logging.info(f"Acceptance rate: {ar}")
    logging.info(f"ESS: {ess(samples[:, 0], cfg.target_density.mu[0], cfg.target_density.std[0])}")

    plot_samples_with_momentum(
        samples,
        target_density=density,
        q_0=0.0,
        q_1=0.0,
        name=cfg.figure_path / Path("hmc_samples.png"),
        ar=ar,
        s=4.0,
        c="red",
        alpha=0.8,
    )


if __name__ == "__main__":
    app.run(main)
