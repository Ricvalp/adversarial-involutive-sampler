import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
from absl import logging
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from torch.utils.data import DataLoader

import wandb
from discriminator_models import create_simple_discriminator, log_plot
from sampling import metropolis_hastings_with_momentum, plot_samples_with_density
from trainers.utils import SamplesDataset, numpy_collate


class Trainer:
    def __init__(
        self,
        cfg,
        density,
        wandb_log,
        checkpoint_dir,
        checkpoint_name,
        seed,
    ):
        self.rng = jax.random.PRNGKey(seed)

        self.density = density
        self.wandb_log = wandb_log
        self.checkpoint_path = os.path.join(
            os.path.join(checkpoint_dir, cfg.target_density.name), checkpoint_name
        )

        self.cfg = cfg

        self.init_model()
        self.create_train_steps()

        # save cfg into checkpoint_path

    def init_model(self):
        discriminator = create_simple_discriminator(
            num_flow_layers=self.cfg.kernel.num_flow_layers,
            num_hidden_flow=self.cfg.kernel.num_hidden,
            num_layers_flow=self.cfg.kernel.num_layers,
            num_layers_psi=self.cfg.discriminator.num_layers_psi,
            num_hidden_psi=self.cfg.discriminator.num_hidden_psi,
            num_layers_eta=self.cfg.discriminator.num_layers_eta,
            num_hidden_eta=self.cfg.discriminator.num_hidden_eta,
            activation=self.cfg.discriminator.activation,
            d=self.cfg.kernel.d,
        )

        self.rng, init_rng, init_points_rng = jax.random.split(self.rng, 3)

        discriminator_params = discriminator.init(
            init_rng, jax.random.normal(init_points_rng, (100, 2 * self.cfg.kernel.d))
        )["params"]

        theta_params = discriminator_params["L"]
        phi_params = discriminator_params["D"]

        L_optimizer = optax.adam(learning_rate=self.cfg.train.kernel_learning_rate)
        discriminator_optimizer = optax.adam(
            learning_rate=self.cfg.train.discriminator_learning_rate
        )

        self.L_state = TrainState.create(
            apply_fn=discriminator.L.apply, params=theta_params, tx=L_optimizer
        )
        self.D_state = TrainState.create(
            apply_fn=discriminator.apply, params=phi_params, tx=discriminator_optimizer
        )

    def create_train_steps(self):
        def maximize_AR_step(L_state, D_state, batch):
            loss_fn = lambda theta_params: AR_loss(theta_params, D_state, batch)
            ar_loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(L_state.params)
            L_state = L_state.apply_gradients(grads=grads)

            return L_state, ar_loss

        self.maximize_AR_step = jax.jit(maximize_AR_step)

        def minimize_adversarial_loss_step(L_state, D_state, batch):
            my_loss = lambda phi_params: adversarial_loss(phi_params, D_state, L_state, batch)
            adv_loss, grads = jax.value_and_grad(my_loss, has_aux=False)(D_state.params)
            D_state = D_state.apply_gradients(grads=grads)

            return D_state, adv_loss

        self.minimize_adversarial_loss_step = jax.jit(minimize_adversarial_loss_step)

    def create_data_loader(self, key, epoch_idx):
        key, subkey = jax.random.split(key)

        samples, _ = self.sample(
            rng=subkey,
            n=self.cfg.train.num_resampling_steps,
            burn_in=self.cfg.train.resampling_burn_in,
            parallel_chains=self.cfg.train.num_resampling_parallel_chains,
            name=None,  # f"samples_in_data_loader_epoch_{epoch_idx}.png",
        )

        dataset = SamplesDataset(np.array(samples))
        self.data_loader = DataLoader(
            dataset, batch_size=self.cfg.train.batch_size, shuffle=True, collate_fn=numpy_collate
        )

        return key

    def train_epoch(self, epoch_idx):
        self.rng = self.create_data_loader(self.rng, epoch_idx)
        for i, batch in enumerate(self.data_loader):
            for _ in range(self.cfg.train.num_AR_steps):
                self.L_state, ar_loss = self.maximize_AR_step(self.L_state, self.D_state, batch)
            for _ in range(self.cfg.train.num_adversarial_steps):
                self.D_state, adv_loss = self.minimize_adversarial_loss_step(
                    self.L_state, self.D_state, batch
                )

            print(f"Epoch: {epoch_idx}, AR loss: {ar_loss}, adversarial loss: {adv_loss}")

            if self.wandb_log:
                wandb.log(
                    {
                        "Epoch: ": epoch_idx,
                        "AR loss": ar_loss,
                        "adversarial loss": adv_loss,
                    }
                )

            if i % self.cfg.log.log_every == 0:
                _, ar = self.sample(
                    rng=self.rng,
                    n=self.cfg.log.num_steps,
                    burn_in=self.cfg.log.burn_in,
                    parallel_chains=self.cfg.log.num_parallel_chains,
                    name=None,  # f"samples_{epoch_idx}.png",
                )
                self.save_model(epoch=epoch_idx, step=i)

                if self.wandb_log:
                    wandb.log({"acceptance rate": ar})

                    fig = log_plot(
                        discriminator_parameters={"params": {"D": self.D_state.params}},
                        num_layers_psi=self.cfg.discriminator.num_layers_psi,
                        num_hidden_psi=self.cfg.discriminator.num_hidden_psi,
                        num_layers_eta=self.cfg.discriminator.num_layers_eta,
                        num_hidden_eta=self.cfg.discriminator.num_hidden_eta,
                        activation=self.cfg.discriminator.activation,
                        d=self.cfg.kernel.d,
                        name="discriminator",
                    )
                    wandb.log({"discriminator": fig})

    def train_model(self):
        for epoch in range(self.cfg.train.num_epochs):
            self.train_epoch(epoch_idx=epoch)

    def sample(self, rng, n, burn_in, parallel_chains, name):
        kernel_fn = jax.jit(lambda x: self.L_state.apply_fn({"params": self.L_state.params}, x))
        logging.info("Sampling...")
        start_time = time.time()
        samples, ar = metropolis_hastings_with_momentum(
            kernel=kernel_fn,
            density=self.density,
            d=self.cfg.kernel.d,
            n=n,
            cov_p=jnp.eye(self.cfg.kernel.d),
            parallel_chains=parallel_chains,
            burn_in=burn_in,
            rng=rng,
        )
        logging.info(f"Sampling took {time.time() - start_time} seconds")

        if name is not None:
            name = self.cfg.figure_path / Path(name)

        fig = plot_samples_with_density(
            samples,
            target_density=self.density,
            q_0=0.0,
            q_1=0.0,
            name=name,
            s=0.5,
            c="red",
            alpha=0.05,
            ar=ar,
        )

        if self.wandb_log:
            wandb.log({f"samples with {parallel_chains} chains": fig})

        return samples, ar

    def save_model(self, epoch, step):
        ckpt = {"L": self.L_state, "D": self.D_state}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(
            os.path.join(self.checkpoint_path, f"{epoch}_{step}"), ckpt, save_args=save_args
        )

        # log cfg into checkpoint_path
        if epoch == 0 and step == 0:
            with open(os.path.join(self.checkpoint_path, "cfg.txt"), "w") as f:
                f.write(str(self.cfg))

    def load_model(self, epoch, step):
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        ckpt = orbax_checkpointer.restore(os.path.join(self.checkpoint_path, f"{epoch}_{step}"))
        self.L_state = ckpt["L"]
        self.D_state = ckpt["D"]

class TrainerLogisticRegression:
    def __init__(
        self,
        cfg,
        density,
        wandb_log,
        checkpoint_dir,
        checkpoint_name,
        seed,
    ):
        self.rng = jax.random.PRNGKey(seed)

        self.density = density
        self.wandb_log = wandb_log
        self.checkpoint_path = os.path.join(
            os.path.join(checkpoint_dir, cfg.target_density.name), checkpoint_name
        )

        self.cfg = cfg

        self.init_model()
        self.create_train_steps()

        # save cfg into checkpoint_path

    def init_model(self):
        discriminator = create_simple_discriminator(
            num_flow_layers=self.cfg.kernel.num_flow_layers,
            num_hidden_flow=self.cfg.kernel.num_hidden,
            num_layers_flow=self.cfg.kernel.num_layers,
            num_layers_psi=self.cfg.discriminator.num_layers_psi,
            num_hidden_psi=self.cfg.discriminator.num_hidden_psi,
            num_layers_eta=self.cfg.discriminator.num_layers_eta,
            num_hidden_eta=self.cfg.discriminator.num_hidden_eta,
            activation=self.cfg.discriminator.activation,
            d=self.cfg.kernel.d,
        )

        self.rng, init_rng, init_points_rng = jax.random.split(self.rng, 3)

        discriminator_params = discriminator.init(
            init_rng, jax.random.normal(init_points_rng, (100, 2 * self.cfg.kernel.d))
        )["params"]

        theta_params = discriminator_params["L"]
        phi_params = discriminator_params["D"]

        L_optimizer = optax.adam(learning_rate=self.cfg.train.kernel_learning_rate)
        discriminator_optimizer = optax.adam(
            learning_rate=self.cfg.train.discriminator_learning_rate
        )

        self.L_state = TrainState.create(
            apply_fn=discriminator.L.apply, params=theta_params, tx=L_optimizer
        )
        self.D_state = TrainState.create(
            apply_fn=discriminator.apply, params=phi_params, tx=discriminator_optimizer
        )

    def create_train_steps(self):
        def maximize_AR_step(L_state, D_state, batch):
            loss_fn = lambda theta_params: AR_loss(theta_params, D_state, batch)
            ar_loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(L_state.params)
            L_state = L_state.apply_gradients(grads=grads)

            return L_state, ar_loss

        self.maximize_AR_step = jax.jit(maximize_AR_step)

        def minimize_adversarial_loss_step(L_state, D_state, batch):
            my_loss = lambda phi_params: adversarial_loss(phi_params, D_state, L_state, batch)
            adv_loss, grads = jax.value_and_grad(my_loss, has_aux=False)(D_state.params)
            D_state = D_state.apply_gradients(grads=grads)

            return D_state, adv_loss

        self.minimize_adversarial_loss_step = jax.jit(minimize_adversarial_loss_step)

    def create_data_loader(self, key, epoch_idx):
        key, subkey = jax.random.split(key)

        samples, _ = self.sample(
            rng=subkey,
            n=self.cfg.train.num_resampling_steps,
            burn_in=self.cfg.train.resampling_burn_in,
            parallel_chains=self.cfg.train.num_resampling_parallel_chains,
            name=None,  # f"samples_in_data_loader_epoch_{epoch_idx}.png",
        )

        dataset = SamplesDataset(np.array(samples))
        self.data_loader = DataLoader(
            dataset, batch_size=self.cfg.train.batch_size, shuffle=True, collate_fn=numpy_collate
        )

        return key

    def train_epoch(self, epoch_idx):
        self.rng = self.create_data_loader(self.rng, epoch_idx)
        for i, batch in enumerate(self.data_loader):
            for _ in range(self.cfg.train.num_AR_steps):
                self.L_state, ar_loss = self.maximize_AR_step(self.L_state, self.D_state, batch)
            for _ in range(self.cfg.train.num_adversarial_steps):
                self.D_state, adv_loss = self.minimize_adversarial_loss_step(
                    self.L_state, self.D_state, batch
                )

            print(f"Epoch: {epoch_idx}, AR loss: {ar_loss}, adversarial loss: {adv_loss}")

            if self.wandb_log:
                wandb.log(
                    {
                        "Epoch: ": epoch_idx,
                        "AR loss": ar_loss,
                        "adversarial loss": adv_loss,
                    }
                )

            if i % self.cfg.log.log_every == 0:
                _, ar = self.sample(
                    rng=self.rng,
                    n=self.cfg.log.num_steps,
                    burn_in=self.cfg.log.burn_in,
                    parallel_chains=self.cfg.log.num_parallel_chains,
                    name=None,  # f"samples_{epoch_idx}.png",
                )
                self.save_model(epoch=epoch_idx, step=i)

                if self.wandb_log:
                    wandb.log({"acceptance rate": ar})

                    fig = log_plot(
                        discriminator_parameters={"params": {"D": self.D_state.params}},
                        num_layers_psi=self.cfg.discriminator.num_layers_psi,
                        num_hidden_psi=self.cfg.discriminator.num_hidden_psi,
                        num_layers_eta=self.cfg.discriminator.num_layers_eta,
                        num_hidden_eta=self.cfg.discriminator.num_hidden_eta,
                        activation=self.cfg.discriminator.activation,
                        d=self.cfg.kernel.d,
                        name="discriminator",
                    )
                    wandb.log({"discriminator": fig})

    def train_model(self):
        for epoch in range(self.cfg.train.num_epochs):
            self.train_epoch(epoch_idx=epoch)

    def sample(self, rng, n, burn_in, parallel_chains, name):
        kernel_fn = jax.jit(lambda x: self.L_state.apply_fn({"params": self.L_state.params}, x))
        logging.info("Sampling...")
        start_time = time.time()
        samples, ar = metropolis_hastings_with_momentum(
            kernel=kernel_fn,
            density=self.density,
            d=self.cfg.kernel.d,
            n=n,
            cov_p=jnp.eye(self.cfg.kernel.d),
            parallel_chains=parallel_chains,
            burn_in=burn_in,
            rng=rng,
        )
        logging.info(f"Sampling took {time.time() - start_time} seconds")

        if name is not None:
            name = self.cfg.figure_path / Path(name)

        fig = plot_samples_with_density(
            samples,
            target_density=self.density,
            q_0=0.0,
            q_1=0.0,
            name=name,
            s=0.5,
            c="red",
            alpha=0.05,
            ar=ar,
        )

        if self.wandb_log:
            wandb.log({f"samples with {parallel_chains} chains": fig})

        return samples, ar

    def save_model(self, epoch, step):
        ckpt = {"L": self.L_state, "D": self.D_state}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(
            os.path.join(self.checkpoint_path, f"{epoch}_{step}"), ckpt, save_args=save_args
        )

        # log cfg into checkpoint_path
        if epoch == 0 and step == 0:
            with open(os.path.join(self.checkpoint_path, "cfg.txt"), "w") as f:
                f.write(str(self.cfg))

    def load_model(self, epoch, step):
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        ckpt = orbax_checkpointer.restore(os.path.join(self.checkpoint_path, f"{epoch}_{step}"))
        self.L_state = ckpt["L"]
        self.D_state = ckpt["D"]





def r(y):
    return 1 / (1 + jnp.exp(-y))


def AR_loss(theta_params, D_state, batch):
    Dx = D_state.apply_fn(
        {
            "params": {
                "L": theta_params,
                "D": D_state.params,
            }
        },
        batch,
    )

    return -(r(Dx)).mean()


def adversarial_loss(phi_params, D_state, L_state, batch):
    Dx = D_state.apply_fn(
        {
            "params": {
                "L": L_state.params,
                "D": phi_params,
            }
        },
        batch,
    )

    return (r(Dx) * jnp.log(r(Dx))).mean()
