"""
====
loss
====

Defines loss functions.

"""

import jax
import jax.numpy as jnp
from jax import grad, jacrev


def scaled_pseudo_huber(delta, r):
    r"""Compute scaled Pseudo-Huber loss.

    :math:`h = \delta \left(\sqrt{1+\left(r/\delta\right)^2} - 1\right)`;
    this is actually :math:`1/\delta` times ``scipy.special.pseudo_huber``,
    and so has slope of one in the linear range.

    Parameters
    ----------
    delta : float
    r : jax.numpy.ndarray

    Return
    -------
    DeviceArray of same length as ``r``.

    >>> h = scaled_pseudo_huber(2., jnp.array([1., 2., 4., 8.]))
    >>> h.round(2)
    DeviceArray([0.24, 0.83, 2.47, 6.25], dtype=float32)
    """
    return delta * (jnp.sqrt(1. + jnp.square(r / delta)) - 1.)
