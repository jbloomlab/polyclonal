"""
Tests for `loss.py`.
"""

import jax
import jax.numpy as jnp
from jax import grad, jacrev

import polyclonal
import polyclonal.loss as loss


def test_pseudo_huber():
    delta = 2.
    r = [1., 2., 4., 8.]
    h, hgrad = polyclonal.Polyclonal._scaled_pseudo_huber(delta, r, True)
    jax_h = loss.scaled_pseudo_huber(delta, jnp.array(r))
    assert jnp.allclose(h, jax_h)

    # Argnums specifies here that we want the gradient WRT the second argument.
    huberish_jac = jacrev(loss.scaled_pseudo_huber, argnums=1)
    jax_hgrad = jnp.diag(huberish_jac(delta, jnp.array(r)))
    assert jnp.allclose(hgrad, jax_hgrad)
