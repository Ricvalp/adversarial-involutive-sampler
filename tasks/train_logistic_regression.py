from absl import app
from ml_collections import config_flags

import jax.numpy as jnp

import wandb
from config import load_cfgs
from trainers import Trainer

from sklearn.datasets import fetch_openml

from config import load_cfgs
from logistic_regression import (
    generate_dataset,
    get_predictions,
    hamiltonian,
    normalize_covariates,
    plot_density_logistic_regression,
    plot_gradients_logistic_regression_density,
    plot_histograms2d_logistic_regression,
    plot_histograms_logistic_regression,
    plot_logistic_regression_samples,
)

_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/config.py")


def main(_):
    cfg = load_cfgs(_TASK_FILE)
    cfg.figure_path.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if cfg.wandb.use:
        # os.environ["WANDB_SILENT"] = "true"
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=cfg)


    heart_data = fetch_openml(name="heart", version=1)
    X_data = heart_data.data.toarray()
    # X_data = normalize_covariates(X_data)
    X = X_data[:, : cfg.dataset.num_covariates]
    X = jnp.concatenate([X, jnp.ones((X.shape[0], 1))], axis=1)
    t_data = heart_data.target
    t = t_data  # [:cfg.dataset.test_split]
    t = (t == 1).astype(int).astype(float)

    # w=jnp.array([-2.5, 1.5, .5])
    # t, X = generate_dataset(
    #     n=25,
    #     w=w,
    #     rng=jax.random.PRNGKey(1)
    # )

    density = lambda x: hamiltonian(x, t, X, inv_sigma=jnp.eye(X.shape[1]) * 0.01)

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
