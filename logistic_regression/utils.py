import matplotlib.pyplot as plt


def plot_logistic_regression_samples(
    samples, i, j, plot=plt.scatter, num_chains=None, name=None, **kwargs
):
    plt.figure(figsize=(10, 10))
    plt.scatter(samples[:, i], samples[:, j], s=1, **kwargs)
    plt.scatter(samples[0, i], samples[0, j], s=10, c="red", marker="x", label="start")
    plt.scatter(samples[-1, i], samples[-1, j], s=10, c="green", marker="x", label="end")

    if num_chains is not None:
        for n in range(num_chains - 1):
            plt.scatter(samples[n, i], samples[n, j], s=10, c="red", marker="x")
            plt.scatter(samples[-n, i], samples[-n, j], s=10, c="green", marker="x")
        plt.scatter(samples[n + 1, i], samples[n + 1, j], s=10, c="red", marker="x", label="start")
        plt.scatter(
            samples[-n - 1, i], samples[-n - 1, j], s=10, c="green", marker="x", label="end"
        )

    plt.xlabel(f"$w_{i}$")
    plt.ylabel(f"$w_{j}$")
    plt.legend(fontsize=20)
    if name is not None:
        plt.savefig(name)
    plt.close()


def plot_histograms2d_logistic_regression(samples, i, j, name=None, **kwargs):
    plt.figure(figsize=(10, 10))
    plt.hist2d(samples[:, i], samples[:, j], bins=100, **kwargs)
    plt.xlabel(f"$w_{i}$")
    plt.ylabel(f"$w_{j}$")
    if name is not None:
        plt.savefig(name)
    plt.close()


def plot_histograms_logistic_regression(samples, i, name=None, **kwargs):
    plt.figure(figsize=(10, 10))
    plt.hist(samples[:, i], bins=100, density=True, **kwargs)
    plt.xlabel(f"$w_{i}$")
    if name is not None:
        plt.savefig(name)
    plt.close()
