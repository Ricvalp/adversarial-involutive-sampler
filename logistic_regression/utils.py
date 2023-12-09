import matplotlib.pyplot as plt


def plot_logistic_regression_samples(samples, i, j, name=None, **kwargs):
    plt.figure(figsize=(10, 10))
    plt.scatter(samples[:, i], samples[:, j], s=1, **kwargs)
    if name is not None:
        plt.savefig(name)
    plt.close()
