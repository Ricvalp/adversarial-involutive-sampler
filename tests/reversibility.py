import jax
import jax.numpy as jnp
from absl.testing import absltest

from discriminator_models import create_simple_discriminator
from kernel_models import create_henon_flow


class TestReversibility(absltest.TestCase):
    def setUp(self):
        num_flow_layers = 5
        num_hidden_flow = 32
        num_layers_flow = 2
        self.d = 2

        self.R = jnp.concatenate(
            jnp.array([[1.0 for _ in range(self.d)], [-1.0 for _ in range(self.d)]])
        )

        self.L = create_henon_flow(
            num_flow_layers=num_flow_layers,
            num_hidden=num_hidden_flow,
            num_layers=num_layers_flow,
            d=self.d,
        )

        self.params = self.L.init(jax.random.PRNGKey(42), jnp.zeros((10, 2 * self.d)))["params"]

    def test_reversibility(self):
        x = jax.random.normal(jax.random.PRNGKey(42), (100, 2 * self.d))
        Lx = self.L.apply({"params": self.params}, x)
        RLx = self.R * Lx
        RLRLx = self.R * self.L.apply({"params": self.params}, RLx)

        self.assertAlmostEqual(jnp.max(jnp.abs(x - RLRLx)), 0.0, places=5)

    def tearDown(self):
        pass
