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


def prox_grad_of_polyclonal(
    poly_abs,
    loss_delta=0.1,
    reg_escape_weight=0.01,
    reg_escape_delta=0.1,
    reg_spread_weight=0.25,
):
    bv_sparse = loss.bv_sparse_of_bmap(poly_abs._binarymaps)
    (matrix_to_mean, coeff_positions) = loss.spread_matrices_of_polyclonal(poly_abs)
    loss_grad = jax.grad(loss.loss, 0)
    args = [
        poly_abs,
        bv_sparse,
        loss_delta,
        reg_escape_weight,
        reg_escape_delta,
        reg_spread_weight,
        matrix_to_mean,
        coeff_positions,
    ]
    return optimization.AccProxGrad(
        lambda p: loss.loss(p, *args),
        lambda p: loss_grad(p, *args),
        zero_function,
        trivial_prox,
        verbose=True,
    )