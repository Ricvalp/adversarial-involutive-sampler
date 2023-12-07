import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from absl import app, logging
from ml_collections import config_flags

import densities
from config import load_cfgs
from densities import plot_density
from kernel_models import create_henon_flow
from sampling import metropolis_hastings, plot_samples, random_walk_kernel

_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/config.py")


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    cfg.figure_path.mkdir(parents=True, exist_ok=True)

    density = getattr(densities, cfg.target_density.name)

    plot_density(density, xlim=6, ylim=6, n=100, name=cfg.figure_path / Path("density.png"))

    kernel_fn = random_walk_kernel(0.1)

    samples = metropolis_hastings(
        kernel_fn,
        density,
        d=cfg.kernel.d,
        parallel_chains=100,
        n=10000,
        burn_in=100,
        seed=cfg.seed,
    )

    plot_samples(
        samples,
        target_density=density,
        d=cfg.sample.d,
        name=cfg.figure_path / Path("samples"),
        s=0.5,
        c="red",
        alpha=0.05,
    )


if __name__ == "__main__":
    app.run(main)
