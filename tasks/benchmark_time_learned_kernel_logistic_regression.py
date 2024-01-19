import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, logging
from ml_collections import config_flags

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config import load_cfgs
from kernel_models import create_henon_flow
from kernel_models.utils import get_params_from_checkpoint
from sampling import metropolis_hastings_with_momentum

import logistic_regression

_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/config.py")


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    cfg.figure_path.mkdir(parents=True, exist_ok=True)

    density = getattr(logistic_regression, cfg.dataset.name)(batch_size=cfg.sample.num_parallel_chains)

    # hmc_samples = np.load(cfg.hmc_sample_dir / Path(f"hmc_samples_{cfg.dataset.name}.npy"))

    checkpoint_path = os.path.join(
        os.path.join(cfg.checkpoint_dir, cfg.dataset.name), cfg.checkpoint_name
    )

    kernel_params, discriminator_params = get_params_from_checkpoint(
        checkpoint_path=checkpoint_path,
        checkpoint_epoch=cfg.checkpoint_epoch,
        checkpoint_step=cfg.checkpoint_step,
    )

    kernel = create_henon_flow(
        num_flow_layers=cfg.kernel.num_flow_layers,
        num_hidden=cfg.kernel.num_hidden,
        num_layers=cfg.kernel.num_layers,
        d=density.dim # cfg.kernel.d,
    )

    kernel_fn = jax.jit(lambda x: kernel.apply(kernel_params, x))

    nums_parallel_chains = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 5096, 5096*2]
    T = []
    for i in nums_parallel_chains:

        density = getattr(logistic_regression, cfg.dataset.name)(batch_size=i)

        samples, ar, t = metropolis_hastings_with_momentum(
            kernel_fn,
            density,
            cov_p=jnp.eye(density.dim),
            d=density.dim,
            parallel_chains=i,
            n=cfg.sample.num_iterations,
            burn_in=cfg.sample.burn_in,
            rng=jax.random.PRNGKey(cfg.seed+i),
            # starting_points=hmc_samples[i+0:],
        )

        T.append(t)

    np.save(cfg.data_for_plots_dir / Path(f"learned_kernel_time_{cfg.dataset.name}.npy"), np.concatenate(np.array([nums_parallel_chains, T])))

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
