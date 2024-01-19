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
    cfg.seed = 42

    cfg.figure_path = pathlib.Path("./figures") / datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg.checkpoint_dir = pathlib.Path("./checkpoints")
    cfg.checkpoint_name = "debug"
    cfg.overwrite = True

    # bootstrap with hmc
    cfg.hmc_sample_dir = pathlib.Path("./hmc_samples")

    # tracing
    cfg.trace_dir = pathlib.Path("./traces")

    # Restore checkpoint
    cfg.checkpoint_epoch = 7
    cfg.checkpoint_step = 0

    # Target density
    cfg.target_density = ConfigDict()
    cfg.target_density.name = "ring"

    # Wandb
    cfg.wandb = ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.project = "adversarial-involutive-sampler-debug"
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
    cfg.train.kernel_learning_rate = 1e-4
    cfg.train.discriminator_learning_rate = 1e-4
    cfg.train.num_resampling_steps = 5000
    cfg.train.num_resampling_parallel_chains = 1
    cfg.train.resampling_burn_in = 1000
    cfg.train.batch_size = 4096
    cfg.train.num_epochs = 100
    cfg.train.num_epochs_hmc_bootstrap = 200
    cfg.train.num_AR_steps = 1
    cfg.train.num_adversarial_steps = 1
    cfg.train.bootstrap_with_hmc = True
    
    # Log
    cfg.log = ConfigDict()
    cfg.log.log_every = 500
    cfg.log.num_steps = 10000
    cfg.log.num_parallel_chains = 2
    cfg.log.burn_in = 100
    cfg.log.samples_to_plot = 5000

    # Dataset
    cfg.dataset = ConfigDict()
    cfg.dataset.name = "Australian"

    if mode == "sample":

        # Sample
        cfg.sample = ConfigDict()
        cfg.sample.d = 2
        cfg.sample.num_parallel_chains = 100
        cfg.sample.num_iterations = 5000 # after burn-in
        cfg.sample.burn_in = 1000

        cfg.sample.average_results_over_trials = 10
        cfg.sample.save_samples = False
        cfg.sample.hmc_sample_dir = pathlib.Path("./hmc_samples") 

        # HMC
        cfg.hmc = ConfigDict()
        cfg.hmc.potential_function_name = "mog6"
        cfg.hmc.num_steps = 40
        cfg.hmc.step_size = 0.1

    return cfg
