from absl import app
from ml_collections import config_flags
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import wandb
from config import load_cfgs
from trainers import TrainerLogisticRegression

from config import load_cfgs
from logistic_regression import (
    get_predictions,
    hamiltonian,
    normalize_covariates,
    Heart,
)


_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/config.py")


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    cfg.figure_path.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if cfg.wandb.use:
        # os.environ["WANDB_SILENT"] = "true"
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=cfg)


    density = Heart(batch_size=cfg.train.num_resampling_parallel_chains)

    # data = np.load(f'data/{cfg.dataset.name}/data.npy')
    # labels = np.load(f'data/{cfg.dataset.name}/labels.npy')
    # X_data = normalize_covariates(data)
    # X = X_data[:, : cfg.dataset.num_covariates]
    # X = jnp.concatenate([X, jnp.ones((X.shape[0], 1))], axis=1)
    # t = labels[:, 0]
    # t = (t == 1).astype(int).astype(float)

    # density = lambda x: hamiltonian(x, t, X, inv_sigma=jnp.eye(X.shape[1]) * 0.01)







    # use two densities because of the fixed batch size !!!






    if cfg.train.bootstrap_with_hmc:
        hmc_samples = np.load(cfg.hmc_sample_dir / Path(f"hmc_samples_{cfg.dataset.name}.npy"))

        trainer = TrainerLogisticRegression(
            cfg=cfg,
            density=density,
            wandb_log=cfg.wandb.use,
            checkpoint_dir=cfg.checkpoint_dir,
            checkpoint_name=cfg.checkpoint_name,
            seed=cfg.seed,
            hmc_samples=hmc_samples,
        )
    
    else:
        trainer = TrainerLogisticRegression(
            cfg=cfg,
            density=density,
            wandb_log=cfg.wandb.use,
            checkpoint_dir=cfg.checkpoint_dir,
            checkpoint_name=cfg.checkpoint_name,
            seed=cfg.seed,
            hmc_samples=None,
        )

    trainer.train_model()


if __name__ == "__main__":
    app.run(main)
