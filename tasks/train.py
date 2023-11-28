import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from absl import app, logging
from flax.training.train_state import TrainState
from ml_collections import config_flags

import densities
import wandb
from config import load_cfgs
from discriminator_models import create_simple_discriminator
from kernel_models import create_henon_flow
from sampling import (
    metropolis_hastings_with_momentum,
    plot_samples,
    plot_samples_with_momentum,
)
from trainers import AR_loss, Trainer, adversarial_loss

_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/config.py")


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    cfg.figure_path.mkdir(parents=True, exist_ok=True)

    wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=cfg)

    density = getattr(densities, cfg.target_density.name)

    trainer = Trainer(cfg=cfg, density=density, wandb_log=cfg.wandb.use, seed=cfg.seed)

    trainer.train_model()


if __name__ == "__main__":
    app.run(main)
