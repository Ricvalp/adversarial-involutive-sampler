from torchvision import datasets, transforms


def get_dataset(name, path):
    if name == "MNIST":
        return datasets.MNIST(path)
    elif name == "CIFAR10":
        return datasets.CIFAR10(path)
    else:
        raise NotImplementedError(f"Dataset {name} not implemented")
