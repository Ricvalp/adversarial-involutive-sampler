import matplotlib.pyplot as plt


"""
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
"""

def plot_logistic_regression_samples(
    samples, plot=plt.scatter, index=0, num_chains=None, name=None, **kwargs
):
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].scatter(samples[:, 0+index], samples[:, 1+index], s=1, **kwargs)
    axs[0, 0].scatter(samples[0, 0+index], samples[0, 1+index], s=10, c="red", marker="x", label="start")
    axs[0, 0].scatter(samples[-1, 0+index], samples[-1, 1+index], s=10, c="green", marker="x", label="end")

    axs[0, 1].scatter(samples[:, 1+index], samples[:, 2+index], s=1, **kwargs)
    axs[0, 1].scatter(samples[0, 1+index], samples[0, 2+index], s=10, c="red", marker="x", label="start")
    axs[0, 1].scatter(samples[-1, 1+index], samples[-1, 2+index], s=10, c="green", marker="x", label="end")

    axs[1, 0].scatter(samples[:, 2+index], samples[:, 3+index], s=1, **kwargs)
    axs[1, 0].scatter(samples[0, 2+index], samples[0, 3+index], s=10, c="red", marker="x", label="start")
    axs[1, 0].scatter(samples[-1, 2+index], samples[-1, 3+index], s=10, c="green", marker="x", label="end")

    axs[1, 1].scatter(samples[:, 3+index], samples[:, 4+index], s=1, **kwargs)
    axs[1, 1].scatter(samples[0, 3+index], samples[0, 4+index], s=10, c="red", marker="x", label="start")
    axs[1, 1].scatter(samples[-1, 3+index], samples[-1, 4+index], s=10, c="green", marker="x", label="end")

    if num_chains is not None:
        for n in range(num_chains - 1):
            axs[0, 0].scatter(samples[n, 0+index], samples[n, 1+index], s=10, c="red", marker="x")
            axs[0, 0].scatter(samples[-n, 0+index], samples[-n, 1+index], s=10, c="green", marker="x")
            axs[0, 1].scatter(samples[n, 1+index], samples[n, 2+index], s=10, c="red", marker="x")
            axs[0, 1].scatter(samples[-n, 1+index], samples[-n, 2+index], s=10, c="green", marker="x")
            axs[1, 0].scatter(samples[n, 2], samples[n, 3+index], s=10, c="red", marker="x")
            axs[1, 0].scatter(samples[-n, 2+index], samples[-n, 3+index], s=10, c="green", marker="x")
            axs[1, 1].scatter(samples[n, 3+index], samples[n, 4+index], s=10, c="red", marker="x")
            axs[1, 1].scatter(samples[-n, 3+index], samples[-n, 4+index], s=10, c="green", marker="x")
        axs[0, 0].scatter(samples[n + 1, 0+index], samples[n + 1, 1+index], s=10, c="red", marker="x", label="start")
        axs[0, 0].scatter(
            samples[-n - 1, 0+index], samples[-n - 1, 1+index], s=10, c="green", marker="x", label="end"
        )
        axs[0, 1].scatter(samples[n + 1, 1+index], samples[n + 1, 2+index], s=10, c="red", marker="x", label="start")
        axs[0, 1].scatter(
            samples[-n - 1, 1+index], samples[-n - 1, 2+index], s=10, c="green", marker="x", label="end"
        )
        axs[1, 0].scatter(samples[n + 1, 2+index], samples[n + 1, 3+index], s=10, c="red", marker="x", label="start")
        axs[1, 0].scatter(
            samples[-n - 1, 2+index], samples[-n - 1, 3+index], s=10, c="green", marker="x", label="end"
        )
        axs[1, 1].scatter(samples[n + 1, 3+index], samples[n + 1, 4+index], s=10, c="red", marker="x", label="start")
        axs[1, 1].scatter(
            samples[-n - 1, 3+index], samples[-n - 1, 4+index], s=10, c="green", marker="x", label="end"
        )

    axs[0, 0].set_xlabel(f"$w_{0}$")
    axs[0, 0].set_ylabel(f"$w_{1}$")
    axs[0, 0].legend(fontsize=20)
    axs[0, 1].set_xlabel(f"$w_{1}$")
    axs[0, 1].set_ylabel(f"$w_{2}$")
    axs[0, 1].legend(fontsize=20)
    axs[1, 0].set_xlabel(f"$w_{2}$")
    axs[1, 0].set_ylabel(f"$w_{3}$")
    axs[1, 0].legend(fontsize=20)
    axs[1, 1].set_xlabel(f"$w_{3}$")
    axs[1, 1].set_ylabel(f"$w_{4}$")
    axs[1, 1].legend(fontsize=20)
    if name is not None:
        plt.savefig(name)
    plt.close()

    return fig



def plot_histograms2d_logistic_regression(samples, index=0 ,name=None, **kwargs):
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    axs[0, 0].hist2d(samples[:, 0+index], samples[:, 1+index], bins=100, **kwargs)
    axs[0, 1].hist2d(samples[:, 1+index], samples[:, 2+index], bins=100, **kwargs)
    axs[1, 0].hist2d(samples[:, 2+index], samples[:, 3+index], bins=100, **kwargs)
    axs[1, 1].hist2d(samples[:, 3+index], samples[:, 4+index], bins=100, **kwargs)

    axs[0, 0].set_xlabel(f"$w_{0}$")
    axs[0, 0].set_ylabel(f"$w_{1}$")
    axs[0, 1].set_xlabel(f"$w_{1}$")
    axs[0, 1].set_ylabel(f"$w_{2}$")
    axs[1, 0].set_xlabel(f"$w_{2}$")
    axs[1, 0].set_ylabel(f"$w_{3}$")
    axs[1, 1].set_xlabel(f"$w_{3}$")
    axs[1, 1].set_ylabel(f"$w_{4}$")
    if name is not None:
        plt.savefig(name)
    plt.close()

    return fig


def plot_histograms_logistic_regression(samples, index=0, name=None, **kwargs):

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0, 0].hist(samples[:, 0+index], bins=100, density=True, **kwargs)
    axs[0, 1].hist(samples[:, 1+index], bins=100, density=True, **kwargs)
    axs[1, 0].hist(samples[:, 2+index], bins=100, density=True, **kwargs)
    axs[1, 1].hist(samples[:, 3+index], bins=100, density=True, **kwargs)

    axs[0, 0].set_xlabel(f"$w_{0}$")
    axs[0, 1].set_xlabel(f"$w_{1}$")
    axs[1, 0].set_xlabel(f"$w_{2}$")
    axs[1, 1].set_xlabel(f"$w_{3}$")
    if name is not None:
        plt.savefig(name)
    plt.close()

    return fig

