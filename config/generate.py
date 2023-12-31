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
    cfg.checkpoint_dir = pathlib.Path("./checkpoints_generative_model")
    cfg.checkpoint_name = "checkpoint_name_generative_model"

    # Restore checkpoint
    cfg.checkpoint_epoch = 32
    cfg.checkpoint_step = 0

    # Dataset
    cfg.dataset = ConfigDict()
    cfg.dataset.name = "MNIST"
    cfg.dataset.path = pathlib.Path("./data")

    # Wandb
    cfg.wandb = ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.project = "adversarial-involutive-generative-model"
    cfg.wandb.entity = "ricvalp"

    # Kernel
    cfg.kernel = ConfigDict()
    cfg.kernel.num_flow_layers = 5
    cfg.kernel.num_layers = 2
    cfg.kernel.num_hidden = 32

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
    cfg.train.batch_size = 512
    cfg.train.num_epochs = 100
    cfg.train.num_AR_steps = 1
    cfg.train.num_adversarial_steps = 1

    # Log
    cfg.log = ConfigDict()
    cfg.log.log_every = 100

    if mode == "sample":
        # Sample
        cfg.sample = ConfigDict()
        cfg.sample.d = 2
        cfg.sample.num_parallel_chains = 1
        cfg.sample.num_iterations = 20  # after burn-in
        cfg.sample.burn_in = 10

    return cfg
