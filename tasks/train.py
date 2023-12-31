import os

from absl import app
from ml_collections import config_flags

import densities
import wandb
from config import load_cfgs
from trainers import Trainer

_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/config.py")


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    cfg.figure_path.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if cfg.wandb.use:
        # os.environ["WANDB_SILENT"] = "true"
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=cfg)

    density = getattr(densities, cfg.target_density.name)

    trainer = Trainer(
        cfg=cfg,
        density=density,
        wandb_log=cfg.wandb.use,
        checkpoint_dir=cfg.checkpoint_dir,
        checkpoint_name=cfg.checkpoint_name,
        seed=cfg.seed,
    )

    trainer.train_model()


if __name__ == "__main__":
    app.run(main)
