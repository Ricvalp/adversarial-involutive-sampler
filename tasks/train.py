import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from absl import app, logging
from ml_collections import config_flags

import densities
from sampling import plot_samples
from kernel_models import create_henon_flow
from discriminator_models import create_simple_discriminator

from sampling import metropolis_hastings_with_momentum, plot_samples_with_momentum

from trainers import Trainer, AR_loss, adversarial_loss
from flax.training.train_state import TrainState
import optax



from config import load_cfgs


_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/config.py")

def main(_):
    
    cfg = load_cfgs(_TASK_FILE)
    cfg.figure_path.mkdir(parents=True, exist_ok=True)

    density = getattr(densities, cfg.target_density.name)

    trainer = Trainer(cfg, density, cfg.seed)

    trainer.train_model()

if __name__ == "__main__":
    app.run(main)






'''
    discriminator = create_simple_discriminator(
        num_flow_layers=cfg.kernel.num_flow_layers,
        num_hidden_flow=cfg.kernel.num_hidden,
        num_layers_flow=cfg.kernel.num_layers,
        num_layers_psi=cfg.discriminator.num_layers_psi,
        num_hidden_psi=cfg.discriminator.num_hidden_psi,
        num_layers_eta=cfg.discriminator.num_layers_eta,
        num_hidden_eta=cfg.discriminator.num_hidden_eta,
        activation=cfg.discriminator.activation,
        d=cfg.kernel.d,
    )

    discriminator_params = discriminator.init(jax.random.PRNGKey(42), jnp.zeros((10, 2 * cfg.kernel.d)))['params']
    
    
    x = jax.random.normal(jax.random.PRNGKey(42), (100, 2 * cfg.kernel.d))
    
    def r(y):
        return 1 / (1 + jnp.exp(-y))

    def adversarial_loss(phi_params):

        Dx = discriminator.apply({'params':
                                {
                                    'L': discriminator_params['L'],
                                    'D': phi_params,
                                    }
                                },
                                x)

        return (r(Dx) * jnp.log(r(Dx))).mean()
    
    loss, grads = jax.value_and_grad(adversarial_loss)(discriminator_params['D'])



    density = getattr(densities, cfg.target_density.name)
    kernel_fn = jax.jit(
        lambda x: discriminator.L.apply({'params': discriminator_params['L']}, x)
    )

    x = metropolis_hastings_with_momentum(
        kernel_fn,
        density,
        cov_p = jnp.eye(cfg.kernel.d),
        d=cfg.kernel.d,
        parallel_chains=100, 
        n=10000,
        burn_in=100,
        rng=jax.random.PRNGKey(cfg.seed),
    )

    plot_samples_with_momentum(
        x,
        target_density=density,
        q_0=0.,
        q_1=0.,
        name=cfg.figure_path / Path("samples.png"),
        s=0.5,
        c="red",
        alpha=0.05
    )
    
    def r(y):
        return 1 / (1 + jnp.exp(-y))

    # def adversarial_loss(phi_params):

    #     Dx = discriminator.apply({'params':
    #                             {
    #                                 'L': discriminator_params['L'],
    #                                 'D': phi_params,
    #                                 }
    #                             },
    #                             x)

    #     return (r(Dx) * jnp.log(r(Dx))).mean()

    def AR_loss(theta_params):

        Dx = discriminator.apply({'params':
                                {
                                    'L': theta_params,
                                    'D': discriminator_params['D'],
                                    }
                                },
                                x)

        return - (r(Dx)).mean()
    
    loss, grads = jax.value_and_grad(AR_loss)(discriminator_params['L'])


    

'''