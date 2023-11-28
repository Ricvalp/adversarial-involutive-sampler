import jax
import jax.numpy as jnp
from absl.testing import absltest

from discriminator_models import create_simple_discriminator
from kernel_models import create_henon_flow


class TestDiscriminator(absltest.TestCase):

    def setUp(self):
        self.d = 2
        num_flow_layers=5
        num_hidden_flow=32
        num_layers_flow=2

        self.R = jnp.concatenate(jnp.array([[1. for _ in range(self.d)], [-1. for _ in range(self.d)]]))

        self.discriminator = create_simple_discriminator(
            num_flow_layers=num_flow_layers,
            num_hidden_flow=num_hidden_flow,
            num_layers_flow=num_layers_flow,
            num_layers_psi=2,
            num_hidden_psi=32,
            num_layers_eta=2,
            num_hidden_eta=32,
            activation="relu",
            d=self.d,
        )

        x = jnp.zeros((10, 2 * self.d))
        self.params = self.discriminator.init(jax.random.PRNGKey(42), x)['params']

        self.L = create_henon_flow(
            num_flow_layers=num_flow_layers,
            num_hidden=num_hidden_flow,
            num_layers=num_layers_flow,
            d=self.d
        )

    def test_discriminator(self):
        
        x = jax.random.normal(jax.random.PRNGKey(42), (100, 2 * self.d))
        ar = self.discriminator.apply({'params': self.params}, x)

        L_params = self.params["L"]
        
        RLx = self.R * self.L.apply({'params': L_params}, x)

        neg_ar = -self.discriminator.apply({'params': self.params}, RLx)

        self.assertAlmostEqual(jnp.max(ar-neg_ar), 0., places=5)

    def tearDown(self):
        pass
