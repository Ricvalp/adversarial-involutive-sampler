import jax.numpy as jnp
import matplotlib.pyplot as plt


def get_hamiltonian_density_image(
    density, xlim_q, ylim_q, xlim_p, ylim_p, q_0=0.0, q_1=1.0, n=100
):
    x = jnp.linspace(-xlim_q, xlim_q, n)
    y = jnp.linspace(-ylim_q, ylim_q, n)
    X_q, Y_q = jnp.meshgrid(x, y)
    x = jnp.linspace(-xlim_p, xlim_p, n)
    y = jnp.linspace(-ylim_p, ylim_p, n)
    X_p, Y_p = jnp.meshgrid(x, y)
    z_q = jnp.concatenate(
        jnp.array(
            [
                jnp.hstack([X_q.reshape(-1, 1), Y_q.reshape(-1, 1)]),
                jnp.hstack([jnp.zeros((n**2, 1)), jnp.zeros((n**2, 1))]),
            ]
        ),
        axis=1,
    )
    z_p = jnp.concatenate(
        jnp.array(
            [
                jnp.hstack([jnp.zeros((n**2, 1)) + q_0, jnp.zeros((n**2, 1)) + q_1]),
                jnp.hstack([X_p.reshape(-1, 1), Y_p.reshape(-1, 1)]),
            ]
        ),
        axis=1,
    )
    Z_q = jnp.exp(density(z_q)).reshape((n, n))
    Z_p = jnp.exp(density(z_p)).reshape((n, n))

    return Z_p, Z_q


def plot_samples_with_momentum(
    samples, target_density, q_0=0.0, q_1=0.0, name=None, ar=0, **kwargs
):
    xlim_q = jnp.max(jnp.abs(samples[:, 0])) + 1.5
    ylim_q = jnp.max(jnp.abs(samples[:, 1])) + 1.5
    xlim_p = jnp.max(jnp.abs(samples[:, 2])) + 1.5
    ylim_p = jnp.max(jnp.abs(samples[:, 3])) + 1.5

    Z_p, Z_q = get_hamiltonian_density_image(
        target_density, xlim_q, ylim_q, xlim_p, ylim_p, q_0=q_0, q_1=q_1, n=100
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    fig.suptitle(f"Acceptance rate: {ar:.3}")
    ax[0].imshow(Z_q, extent=(-xlim_q, xlim_q, -ylim_q, ylim_q), origin="lower", cmap="viridis")
    ax[0].scatter(samples[:, 0], samples[:, 1], **kwargs)
    ax[0].set_title("q")
    ax[0].set_xlabel("q1")
    ax[0].set_ylabel("q2")
    ax[1].imshow(Z_p, extent=(-xlim_p, xlim_p, -ylim_p, ylim_p), origin="lower", cmap="viridis")
    ax[1].scatter(samples[:, 2], samples[:, 3], **kwargs)
    ax[1].set_title("p")
    ax[1].set_xlabel("p1")
    ax[1].set_ylabel("p2")

    if name is not None:
        plt.savefig(name)
    plt.show()

    return fig


def get_density_image(density, xlim, ylim, n=100):
    x = jnp.linspace(-xlim, xlim, n)
    y = jnp.linspace(-ylim, ylim, n)
    X, Y = jnp.meshgrid(x, y)
    Z = jnp.exp(density(jnp.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])))
    Z = Z.reshape(X.shape)

    return Z


def plot_samples(samples, target_density, name=None, **kwargs):
    xlim = jnp.max(jnp.abs(samples[:, 0])) + 1.5
    ylim = jnp.max(jnp.abs(samples[:, 1])) + 1.5

    Z = get_density_image(target_density, xlim, ylim, n=100)

    plt.imshow(Z, extent=(-xlim, xlim, -ylim, ylim), origin="lower", cmap="viridis")
    plt.scatter(samples[:, 0], samples[:, 1], **kwargs)
    plt.xlabel("x1")
    plt.ylabel("x2")

    if name is not None:
        plt.savefig(name)
    plt.show()
