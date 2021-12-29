"""
====
fit
====

Defines fitting functions for the loss function in JAX.

"""

import jax

import polyclonal.loss as loss
import polyclonal.optimization as optimization


def zero_function(params):
    return 0.0


def trivial_prox(params, t):
    return params


def prox_grad_of_polyclonal(poly_abs, delta):
    exact_bv_sparse = loss.bv_sparse_of_bmap(poly_abs._binarymaps)
    loss_grad = jax.grad(loss.loss, 0)
    return optimization.AccProxGrad(
        lambda p: loss.loss(p, poly_abs, exact_bv_sparse, delta),
        lambda p: loss_grad(p, poly_abs, exact_bv_sparse, delta),
        zero_function,
        trivial_prox,
        verbose=True,
    )
