from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax import random


class ImageFlow(nn.Module):
    # R : jnp.array

    def setup(self):
        flow_layers_before_split = []

        # vardeq_layers = [CouplingLayer(network=GatedConvNet(c_out=2, c_hidden=16),
        #                                mask=create_checkerboard_mask(h=28, w=28, invert=(i%2==1)),
        #                                c_in=1) for i in range(4)]

        # flow_layers_before_split += [Dequantization()]

        flow_layers_before_split += [
            CouplingLayer(
                network=GatedConvNet(c_out=2, c_hidden=32),
                mask=create_checkerboard_mask(h=28, w=28, invert=(i % 2 == 1)),
                c_in=1,
            )
            for i in range(2)
        ]
        flow_layers_before_split += [SqueezeFlow()]
        for i in range(2):
            flow_layers_before_split += [
                CouplingLayer(
                    network=GatedConvNet(c_out=8, c_hidden=48),
                    mask=create_channel_mask(c_in=4, invert=(i % 2 == 1)),
                    c_in=4,
                )
            ]

        split = SplitFlow()

        flow_layers_after_split = []
        flow_layers_after_split += [SqueezeFlow()]
        for i in range(4):
            flow_layers_after_split += [
                CouplingLayer(
                    network=GatedConvNet(c_out=16, c_hidden=64),
                    mask=create_channel_mask(c_in=8, invert=(i % 2 == 1)),
                    c_in=8,
                )
            ]
        squeeze = SqueezeFlow()

        self.flow_layers_before_split = flow_layers_before_split
        self.split = split
        self.squeeze = squeeze
        self.flow_layers_after_split = flow_layers_after_split

    def __call__(self, z, rng):
        # forward
        for flow in self.flow_layers_before_split:
            z, rng = flow(z, rng, reverse=False)
            print("flow: ", z.shape)

        (z, z_split), rng = self.split(z, rng, reverse=False)
        print("split: ", z.shape)

        for flow in self.flow_layers_after_split:
            z, rng = flow(z, rng, reverse=False)
            print("flow: ", z.shape)

        z, rng = self.squeeze(z, rng, reverse=True)
        print("squeeze: ", z.shape)
        z = jnp.concatenate([z, z_split], axis=-1)
        print("concatenate: ", z.shape)
        z, rng = self.squeeze(z, rng, reverse=True)
        print("squeeze: ", z.shape)

        # inverse
        # z = self.R * z

        z, rng = self.squeeze(z, rng, reverse=False)
        z, z_split = jnp.split(z, 2, axis=-1)
        z, rng = self.squeeze(z, rng, reverse=False)

        for flow in reversed(self.flow_layers_after_split):
            z, rng = flow(z, rng, reverse=True)

        z = jnp.concatenate([z, z_split], axis=-1)

        for flow in reversed(self.flow_layers_before_split):
            z, rng = flow(z, rng, reverse=True)

        return z, rng


class CouplingLayer(nn.Module):
    network: nn.Module
    mask: jnp.array
    c_in: int

    def setup(self):
        self.scaling_factor = self.param("scaling_factor", nn.initializers.zeros, (self.c_in,))

    def __call__(self, z, rng, reverse=False):
        z_in = z * self.mask
        nn_out = self.network(z_in)
        s, t = jnp.split(nn_out, 2, axis=-1)

        s_fac = jnp.exp(self.scaling_factor).reshape(1, 1, 1, -1)
        s = nn.tanh(s / s_fac) * s_fac

        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        if not reverse:
            z = (z + t) * jnp.exp(s)
        else:
            z = (z * jnp.exp(-s)) - t

        return z, rng


class Dequantization(nn.Module):
    alpha: float = (
        1e-5  # Small constant that is used to scale the original input for numerical stability.
    )
    quants: int = 256  # Number of possible discrete values (usually 256 for 8-bit image)

    def __call__(self, z, rng, reverse=False):
        if not reverse:
            z, rng = self.dequant(z, rng)
            z = self.sigmoid(z, reverse=True)
        else:
            z = self.sigmoid(z, reverse=False)
            z = z * self.quants
            z = jnp.floor(z)
            z = jax.lax.clamp(min=0.0, x=z, max=self.quants - 1.0).astype(jnp.int32)
        return z, rng

    def sigmoid(self, z, reverse=False):
        # Applies an invertible sigmoid transformation
        if not reverse:
            z = nn.sigmoid(z)
            z = (z - 0.5 * self.alpha) / (1 - self.alpha)
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha  # Scale to prevent boundaries 0 and 1
            z = jnp.log(z) - jnp.log(1 - z)
        return z

    def dequant(self, z, rng):
        # Transform discrete values to continuous volumes
        z = z.astype(jnp.float32)
        rng, uniform_rng = random.split(rng)
        z = z + random.uniform(uniform_rng, z.shape)
        z = z / self.quants
        return z, rng


class ConcatELU(nn.Module):
    """Activation function that applies ELU in both direction (inverted and plain).

    Allows non-linearity while providing strong gradients for any input (important for final
    convolution)
    """

    def __call__(self, x):
        return jnp.concatenate([nn.elu(x), nn.elu(-x)], axis=-1)


class GatedConv(nn.Module):
    """This module applies a two-layer convolutional ResNet block with input gate."""

    c_in: int  # Number of input channels
    c_hidden: int  # Number of hidden dimensions

    @nn.compact
    def __call__(self, x):
        out = nn.Sequential(
            [
                ConcatELU(),
                nn.Conv(self.c_hidden, kernel_size=(3, 3)),
                ConcatELU(),
                nn.Conv(2 * self.c_in, kernel_size=(1, 1)),
            ]
        )(x)
        val, gate = jnp.split(out, 2, axis=-1)
        return x + val * nn.sigmoid(gate)


class GatedConvNet(nn.Module):
    c_hidden: int  # Number of hidden dimensions to use within the network
    c_out: int  # Number of output channels
    num_layers: int = 3  # Number of gated ResNet blocks to apply

    def setup(self):
        layers = []
        layers += [nn.Conv(self.c_hidden, kernel_size=(3, 3))]
        for layer_index in range(self.num_layers):
            layers += [GatedConv(self.c_hidden, self.c_hidden), nn.LayerNorm()]
        layers += [
            ConcatELU(),
            nn.Conv(self.c_out, kernel_size=(3, 3), kernel_init=nn.initializers.zeros),
        ]
        self.nn = nn.Sequential(layers)

    def __call__(self, x):
        return self.nn(x)


class SqueezeFlow(nn.Module):
    def __call__(self, z, rng, reverse=False):
        B, H, W, C = z.shape
        if not reverse:
            # Forward direction: H x W x C => H/2 x W/2 x 4C
            z = z.reshape(B, H // 2, 2, W // 2, 2, C)
            z = z.transpose((0, 1, 3, 2, 4, 5))
            z = z.reshape(B, H // 2, W // 2, 4 * C)
        else:
            # Reverse direction: H/2 x W/2 x 4C => H x W x C
            z = z.reshape(B, H, W, 2, 2, C // 4)
            z = z.transpose((0, 1, 3, 2, 4, 5))
            z = z.reshape(B, H * 2, W * 2, C // 4)
        return z, rng


class SplitFlow(nn.Module):
    def __call__(self, z, rng, z_split=None, reverse=False):
        if not reverse:
            z, z_split = jnp.split(z, 2, axis=-1)
        else:
            z = jnp.concatenate([z, z_split], axis=-1)
        return (z, z_split), rng


def create_checkerboard_mask(h, w, invert=False):
    x, y = jnp.arange(h, dtype=jnp.int32), jnp.arange(w, dtype=jnp.int32)
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    mask = jnp.fmod(xx + yy, 2)
    mask = mask.astype(jnp.float32).reshape(1, h, w, 1)
    if invert:
        mask = 1 - mask
    return mask


def create_momentum_checkerboard_mask(h, w, invert=False):
    mask = create_checkerboard_mask(h, w, invert=invert)
    mask = jnp.concatenate([mask, 1 - mask], axis=-1)
    return mask


def create_channel_mask(c_in, invert=False):
    mask = jnp.concatenate(
        [
            jnp.ones((c_in // 2,), dtype=jnp.float32),
            jnp.zeros((c_in - c_in // 2,), dtype=jnp.float32),
        ]
    )
    mask = mask.reshape(1, 1, 1, c_in)
    if invert:
        mask = 1 - mask
    return mask


def create_multiscale_flow():
    return ImageFlow()


import math

import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


# Transformations applied on each image => bring them into a numpy array
# Note that we keep them in the range 0-255 (integers)
def image_to_numpy(img):
    img = np.array(img, dtype=np.int32)
    img = img[..., None]  # Make image [28, 28, 1]
    return img


# We need to stack the batch elements
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def show_imgs(imgs, title=None, name=None, row_size=4):
    # Form a grid of pictures (we use max. 8 columns)
    imgs = np.copy(jax.device_get(imgs))
    num_imgs = imgs.shape[0]
    is_int = imgs.dtype == np.int32
    nrow = min(num_imgs, row_size)
    ncol = int(math.ceil(num_imgs / nrow))
    imgs_torch = torch.from_numpy(imgs).permute(0, 3, 1, 2)
    imgs = torchvision.utils.make_grid(imgs_torch, nrow=nrow, pad_value=128 if is_int else 0.5)
    np_imgs = imgs.cpu().numpy()
    # Plot the grid
    plt.figure(figsize=(1.5 * nrow, 1.5 * ncol))
    plt.imshow(np.transpose(np_imgs, (1, 2, 0)), interpolation="nearest")
    plt.axis("off")
    if title is not None:
        plt.title(title)
    if name is not None:
        plt.savefig(name, bbox_inches="tight")
    plt.show()
    plt.close()


model = ImageFlow()
rng = random.PRNGKey(0)
rng, init_rng = random.split(rng)

# Loading the training dataset. We need to split it into a training and validation part
# train_dataset = MNIST(root= "../../data", train=True, transform=image_to_numpy, download=True)
# train_set, val_set = torch.utils.data.random_split(
#     train_dataset, [50000, 10000], generator=torch.Generator().manual_seed(42)
# )
# train_loader = data.DataLoader(
#     train_set,
#     batch_size=16,
#     shuffle=True,
#     drop_last=True,
#     collate_fn=numpy_collate,
#     num_workers=8,
#     persistent_workers=True,
# )

# batch = next(iter(train_loader))[0]

# batch = jax.random.normal(rng, (16, 28, 28, 1))
# params = model.init(init_rng, batch, rng)

# y = model.apply(params, batch, rng)[0]
# print(y.shape)
# print(((y - batch) ** 2).mean())
