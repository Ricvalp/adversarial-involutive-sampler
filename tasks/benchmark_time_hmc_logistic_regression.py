from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, logging
from ml_collections import config_flags

from config import load_cfgs

import logistic_regression
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

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import logistic_regression.statistics as statistics
from sampling.metrics import ess, gelman_rubin_r, effective_sample_size
from sampling import hmc

# TensorBoard
import tensorflow as tf
import datetime


_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/config.py")


def main(_):

    cfg = load_cfgs(_TASK_FILE)
    cfg.figure_path.mkdir(parents=True, exist_ok=True)

    # nums_parallel_chains = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 5096, 5096*2]
    nums_parallel_chains = [2048]

    T = []
    for i in nums_parallel_chains:

        density = getattr(logistic_regression, cfg.dataset.name)(batch_size=i)
        grad_potential_fn = density.get_grad_energy_fn()

        samples, ar, t = hmc(
            density=density,
            grad_potential_fn=grad_potential_fn,
            cov_p=jnp.eye(density.dim)* 1.,
            d=density.dim,
            parallel_chains=i,
            num_steps=cfg.hmc.num_steps,
            step_size=cfg.hmc.step_size,
            n=cfg.sample.num_iterations,
            burn_in=cfg.sample.burn_in,
            initial_std=.1,
            rng=jax.random.PRNGKey(cfg.seed + i),
            trace_dir=cfg.trace_dir
        )

        T.append(t)


    np.save(cfg.data_for_plots_dir / Path(f"hmc_time_{cfg.dataset.name}.npy"), np.concatenate(np.array([nums_parallel_chains, T])))

    data = {
        'x': nums_parallel_chains,
        'y': T
        }

    df = pd.DataFrame(data)

    sns.set(style="whitegrid") 
    sns.lineplot(x='x', y='y', data=df, palette='viridis', alpha=0.7)
    sns.set(style="whitegrid", rc={"grid.alpha": 0.4})

    plt.title('Your Title')
    plt.xlabel('Parallel chains')
    plt.ylabel('Time')
    plt.xscale('log')
    plt.savefig('time.png')

    assert True


if __name__ == "__main__":
    app.run(main)
