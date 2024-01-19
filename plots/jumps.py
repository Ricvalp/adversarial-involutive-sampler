import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os
import jax.numpy as jnp
import densities

density_name = "mog2"

figures_path = "./plots/jumps_hmc_" + density_name + ".png"
data_path = Path("./plots/data_for_plots")

samples = np.load(data_path / Path("jumps_hmc_" + density_name + ".npy"))

def plot_hamiltonian_density_only_q(
    density, samples, xlim_q, ylim_q,  n=100, t=100, name=None
):

    x = jnp.linspace(-xlim_q, xlim_q, n)
    y = jnp.linspace(-ylim_q, ylim_q, n)
    X_q, Y_q = jnp.meshgrid(x, y)
    z_q = jnp.concatenate(
        jnp.array(
            [
                jnp.hstack([X_q.reshape(-1, 1), Y_q.reshape(-1, 1)]),
                jnp.hstack([jnp.zeros((n**2, 1)), jnp.zeros((n**2, 1))]),
            ]
        ),
        axis=1,
    )
    Z_q = jnp.exp(-density(z_q)).reshape((n, n))

    fig = plt.figure(figsize=(5, 5))
    im_q = plt.imshow(
        Z_q, extent=(-xlim_q, xlim_q, -ylim_q, ylim_q), origin="lower", cmap="viridis"
    )
    # plt.xlabel(r'$q_1$', fontsize=20)
    # plt.ylabel(r'$q_2$', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.plot(samples[:t, 0], samples[:t, 1], color='red', markersize=1.5, alpha=0.9, linewidth=.8)
    plt.xlim(-xlim_q, xlim_q)
    plt.ylim(-ylim_q, ylim_q)

    if name is not None:
        plt.savefig(name)
    plt.show()
    

plot_hamiltonian_density_only_q(
    getattr(densities, "hamiltonian_" + density_name),
    samples,
    6.5,
    6.5,
    200,
    t=1000,
    name=figures_path
    )