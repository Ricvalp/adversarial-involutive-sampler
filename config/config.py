import pathlib
from datetime import datetime
from typing import Literal

from absl import logging
from ml_collections import ConfigDict


def get_config(mode: Literal["train", "sample"] = None):
    if mode is None:
        mode = "train"
        logging.info(f"No mode provided, using '{mode}' as default")

    cfg = ConfigDict()
    cfg.seed = 43

    cfg.figure_path = pathlib.Path("./figures") / datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg.checkpoint_path = pathlib.Path("./checkpoints")
    cfg.checkpoint_name = "checkpoint_name"

    # Target density
    cfg.target_density = ConfigDict()
    cfg.target_density.name = "hamiltonian_mog2"

    # Wandb
    cfg.wandb = ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.project = "adversarial-involutive-sampler"
    cfg.wandb.entity = "ricvalp"

    # Kernel
    cfg.kernel = ConfigDict()
    cfg.kernel.num_flow_layers = 5
    cfg.kernel.num_layers = 2
    cfg.kernel.num_hidden = 32
    cfg.kernel.d = 2

    # Discriminator
    cfg.discriminator = ConfigDict()
    cfg.discriminator.num_layers_psi = 3
    cfg.discriminator.num_hidden_psi = 32
    cfg.discriminator.num_layers_eta = 3
    cfg.discriminator.num_hidden_eta = 32
    cfg.discriminator.activation = "relu"

    # Train
    cfg.train = ConfigDict()
    cfg.train.kernel_learning_rate = 1e-3
    cfg.train.discriminator_learning_rate = 1e-3
    cfg.train.num_resampling_steps = 1000
    cfg.train.num_resampling_parallel_chains = 100
    cfg.train.resampling_burn_in = 100
    cfg.train.batch_size = 4096
    cfg.train.num_epochs = 100
    cfg.train.num_AR_steps = 1
    cfg.train.num_adversarial_steps = 1

    # Log
    cfg.log = ConfigDict()
    cfg.log.plot_every = 20
    cfg.log.num_steps = 10000
    cfg.log.num_parallel_chains = 2
    cfg.log.burn_in = 100

    # Dataset
    cfg.dataset = ConfigDict()

    if mode == "sample":
        cfg.sample = ConfigDict()

    return cfg
