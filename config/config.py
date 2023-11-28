import pathlib
from datetime import datetime
from typing import Literal, Type

from absl import logging
from ml_collections import ConfigDict, config_dict


def get_config(mode: Literal["sample", "train"] = None):
    if mode is None:
        mode = "sample"
        logging.info(f"No mode provided, using '{mode}' as default")

    cfg = ConfigDict()
    cfg.seed = 43

    cfg.figure_path = pathlib.Path("./figures") / datetime.now().strftime("%Y%m%d-%H%M%S")

    # Target density
    cfg.target_density = ConfigDict()
    cfg.target_density.name = "hamiltonian_mog2"

    # Wandb
    cfg.wandb = ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.project = "adversarial-involutive-sampler"
    cfg.wandb.entity = "ricvalp"

    # Model
    cfg.kernel = ConfigDict()
    cfg.kernel.num_flow_layers = 5
    cfg.kernel.num_layers = 2
    cfg.kernel.num_hidden = 32
    cfg.kernel.d = 2

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
    cfg.train.batch_size = 1024
    cfg.train.num_epochs = 2
    cfg.train.num_AR_steps = 2
    cfg.train.num_adversarial_steps = 1

    cfg.log = ConfigDict()
    cfg.log.plot_every = 20
    cfg.log.num_steps = 1000
    cfg.log.num_parallel_chains = 100
    cfg.log.burn_in = 100

    # Dataset
    cfg.dataset = ConfigDict()

    if mode == "train":
        cfg.sample = ConfigDict()

    return cfg
