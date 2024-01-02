import os

import jax.numpy as jnp
from absl import app
from ml_collections import config_flags
from sklearn.datasets import fetch_openml

import densities
import wandb
from config import load_cfgs
from logistic_regression import (
    hamiltonian_logistic_regression,
    normalize_covariates,
    plot_histograms2d_logistic_regression,
    plot_histograms_logistic_regression,
    plot_logistic_regression_samples,
)
from trainers import Trainer

_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/config.py")


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    cfg.figure_path.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    heart_data = fetch_openml(name="heart", version=1)
    X_data = heart_data.data.toarray()
    X_data = normalize_covariates(X_data)
    X = X_data[:, :3]
    X = jnp.concatenate([X, jnp.ones((X.shape[0], 1))], axis=1)
    t = heart_data.target

    density = lambda x: hamiltonian_logistic_regression(
        x, t, X, cov=jnp.eye(X.shape[1]) * 100, cov_p=jnp.eye(X.shape[1])
    )

    wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=cfg)

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
