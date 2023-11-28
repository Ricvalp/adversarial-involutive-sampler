import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from absl import app, logging
from ml_collections import config_flags

import densities
from densities import plot_hamiltonian_density
from sampling import metropolis_hastings_with_momentum, plot_samples_with_momentum
from kernel_models import create_henon_flow

from config import load_cfgs


_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/config.py")

def main(_):
    cfg = load_cfgs(_TASK_FILE)
    cfg.figure_path.mkdir(parents=True, exist_ok=True)

    density = getattr(densities, cfg.target_density.name)

    kernel = create_henon_flow(
        num_layers_flow=cfg.kernel.num_layers_flow,
        num_hidden=cfg.kernel.num_hidden,
        num_layers=cfg.kernel.num_layers,
        d=cfg.kernel.d,
        )
    
    kernel_params = kernel.init(jax.random.PRNGKey(42), jnp.zeros((10, cfg.kernel.d * 2)))
    kernel_fn = jax.jit(
        lambda x: kernel.apply(kernel_params, x)
    )

    plot_hamiltonian_density(
        density,
        xlim_q=6,
        ylim_q=6,
        xlim_p=6,
        ylim_p=6,
        n=100,
        q_0=0.,
        q_1=0.,
        name= cfg.figure_path / Path("hamiltonian_density.png")
    )
    
    samples = metropolis_hastings_with_momentum(
        kernel_fn,
        density,
        cov_p = jnp.eye(cfg.kernel.d),
        d=cfg.kernel.d,
        parallel_chains=100, 
        n=10000,
        burn_in=100,
        rng=jax.random.PRNGKey(cfg.seed),
    )

    plot_samples_with_momentum(
        samples,
        target_density=density,
        q_0=0.,
        q_1=0.,
        name=cfg.figure_path / Path("samples.png"),
        s=0.5,
        c="red",
        alpha=0.05
    )

if __name__ == "__main__":
    app.run(main)
