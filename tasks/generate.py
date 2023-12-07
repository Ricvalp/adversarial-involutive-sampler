import os

from absl import app
from ml_collections import config_flags

import wandb
from config import load_cfgs
from generative_modelling import get_dataset
from trainers import GenerativeTrainer

_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/generate.py")


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    cfg.figure_path.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset(cfg.dataset.name, cfg.dataset.path)

    checkpoint_path = os.path.join(
        os.path.join(cfg.checkpoint_dir, cfg.target_density.name), cfg.checkpoint_name
    )


if __name__ == "__main__":
    app.run(main)
