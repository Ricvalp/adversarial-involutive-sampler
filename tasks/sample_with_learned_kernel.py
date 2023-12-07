import os
from pathlib import Path

import jax
import jax.numpy as jnp
from absl import app, logging
from ml_collections import config_flags

import densities
from config import load_cfgs
from densities import plot_hamiltonian_density
from discriminator_models import get_discriminator_function, plot_discriminator
from kernel_models import create_henon_flow
from kernel_models.utils import get_params_from_checkpoint
from sampling import metropolis_hastings_with_momentum, plot_samples_with_momentum
from sampling.metrics import ess

_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/config.py")


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    cfg.figure_path.mkdir(parents=True, exist_ok=True)

    density = getattr(densities, cfg.target_density.name)

    checkpoint_path = os.path.join(
        os.path.join(cfg.checkpoint_dir, cfg.target_density.name), cfg.checkpoint_name
    )

    kernel_params, discriminator_params = get_params_from_checkpoint(
        checkpoint_path=checkpoint_path,
        checkpoint_epoch=cfg.checkpoint_epoch,
        checkpoint_step=cfg.checkpoint_step,
    )

    discriminator_fn = get_discriminator_function(
        discriminator_parameters=discriminator_params,
        num_layers_psi=cfg.discriminator.num_layers_psi,
        num_hidden_psi=cfg.discriminator.num_hidden_psi,
        num_layers_eta=cfg.discriminator.num_layers_eta,
        num_hidden_eta=cfg.discriminator.num_hidden_eta,
        activation=cfg.discriminator.activation,
        d=cfg.kernel.d,
    )

    plot_discriminator(
        discriminator_fn,
        xlim_q=6,
        ylim_q=6,
        xlim_p=6,
        ylim_p=6,
        n=100,
        x_0=jnp.array([0.0, 0.0]),
        p_0=0.0,
        p_1=0.0,
        name=cfg.figure_path / Path("discriminator.png"),
    )

    kernel = create_henon_flow(
        num_flow_layers=cfg.kernel.num_flow_layers,
        num_hidden=cfg.kernel.num_hidden,
        num_layers=cfg.kernel.num_layers,
        d=cfg.kernel.d,
    )

    kernel_fn = jax.jit(lambda x: kernel.apply(kernel_params, x))

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

    samples, ar = metropolis_hastings_with_momentum(
        kernel_fn,
        density,
        cov_p=jnp.eye(cfg.kernel.d),
        d=cfg.kernel.d,
        parallel_chains=cfg.sample.num_parallel_chains,
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
        name=cfg.figure_path / Path("samples.png"),
        ar=ar,
        s=4.0,
        c="red",
        alpha=0.8,
    )


if __name__ == "__main__":
    app.run(main)
