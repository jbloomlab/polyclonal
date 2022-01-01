"""
====
loss
====

Defines loss functions via JAX.

"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental import sparse

import numpy as np


# Make JAX use double precision numbers.
jax.config.update("jax_enable_x64", True)


@jit
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
    DeviceArray([0.24, 0.83, 2.47, 6.25], dtype=float64)
    """
    return delta * (jnp.sqrt(1.0 + jnp.square(r / delta)) - 1.0)


# TODO make more efficient?
def bv_sparse_of_bmap(bmap):
    return sparse.BCOO.fromdense(jnp.array(bmap.binary_variants.todense()))


def bv_sparses_of_polyclonal(poly_abs):
    if poly_abs._one_binarymap:
        return bv_sparse_of_bmap(poly_abs._binarymaps)
    else:
        return [bv_sparse_of_bmap(bmap) for bmap in poly_abs._binarymaps]


@partial(jit, static_argnames=["poly_abs"])
def a_beta_from_params(params, poly_abs):
    """Vector of activities and MxE matrix of betas from params vector."""
    n_epitopes = len(poly_abs.epitopes)
    n_mutations = len(poly_abs.mutations)
    params_len = n_epitopes * (1 + n_mutations)
    if params.shape != (params_len,):
        raise ValueError(f"invalid params.shape={params.shape}")
    a = params[:n_epitopes]
    beta = params[n_epitopes:].reshape(n_mutations, n_epitopes)
    assert a.shape == (n_epitopes,)
    assert beta.shape == (n_mutations, n_epitopes)
    return (a, beta)


def spread_matrices_of_polyclonal(poly_abs):
    """
    Builds matrices we can use to do the regularization on spread using matrix math.
    """
    n_mutations = len(poly_abs.mutations)
    # Let's make a matrix, coeff_positions_np, that describes where the betas are for
    # the various sites. It will be of size (number of sites) x (number of betas). It
    # will have a 1 if the given beta is a coefficient for a given site.
    coeff_positions_np = np.zeros((len(poly_abs._binary_sites), n_mutations))
    for row, index_array in zip(coeff_positions_np, poly_abs._binary_sites.values()):
        row[index_array[0]] = 1.0
    # We can turn this into a matrix that will allow us to calculate
    # per-site-per-epitope means by matrix multiplication.
    matrix_to_mean_np = coeff_positions_np / coeff_positions_np.sum(axis=1)[:, None]
    matrix_to_mean = sparse.BCOO.fromdense(jnp.array(matrix_to_mean_np, copy=False))
    # We'd like to use the coeff_positions_np matrix to go from a site-wise view back to
    # a beta-wise view, and for that we transpose.
    coeff_positions = sparse.BCOO.fromdense(
        jnp.array(coeff_positions_np, copy=False)
    ).transpose()
    return (matrix_to_mean, coeff_positions)


@partial(jit, static_argnames=["matrix_to_mean", "coeff_positions", "weight"])
def spread_penalty(beta, matrix_to_mean, coeff_positions, weight):
    """
    Return the sum of the per-site squared deviation of the coefficients from their
    mean.
    """
    to_penalize = beta - coeff_positions @ (matrix_to_mean @ beta)
    return weight * (matrix_to_mean @ (to_penalize ** 2)).sum()


@partial(
    jit, static_argnames=["poly_abs", "matrix_to_mean", "coeff_positions", "weight"]
)
def spread_penalty_of_params(params, poly_abs, matrix_to_mean, coeff_positions, weight):
    """
    As above, but parameterized in terms of params.
    """
    _, beta = a_beta_from_params(params, poly_abs)
    return spread_penalty(beta, matrix_to_mean, coeff_positions, weight)


@partial(jit, static_argnames=["poly_abs", "bv_sparse", "which_cs"])
def compute_pv(params, poly_abs, bv_sparse, which_cs):
    """
    If `which_cs` is None, then there is one binarymap. If there are multiple then it's
    the index of _cs that we want.
    """
    a, beta = a_beta_from_params(params, poly_abs)
    n_epitopes = len(poly_abs.epitopes)
    n_variants = bv_sparse.shape[0]
    if which_cs is None:
        cs = poly_abs._cs
    else:
        cs = poly_abs._cs[which_cs]
    phi_e_v = bv_sparse @ beta - a
    assert phi_e_v.shape == (n_variants, n_epitopes)
    exp_minus_phi_e_v = jnp.exp(-phi_e_v)
    # Using tensordot as a replacement for np.multiply.outer, which doesn't
    # exist in JAX. See
    # https://numpy.org/doc/stable/reference/generated/numpy.ufunc.outer.html#numpy.ufunc.outer
    U_v_e_c = 1.0 / (1.0 + jnp.tensordot(exp_minus_phi_e_v, cs, axes=((), ())))
    assert U_v_e_c.shape == (n_variants, n_epitopes, len(cs))
    n_vc = n_variants * len(cs)
    U_vc_e = jnp.moveaxis(U_v_e_c, 1, 2).reshape(n_vc, n_epitopes, order="F")
    assert U_vc_e.shape == (n_vc, n_epitopes)
    p_vc = U_vc_e.prod(axis=1)
    assert p_vc.shape == (n_vc,)
    return p_vc


@partial(jit, static_argnames=["poly_abs", "bv_sparse", "delta"])
def loss(params, poly_abs, bv_sparse, delta):
    pred_pvs = compute_pv(params, poly_abs, bv_sparse, None)
    assert pred_pvs.shape == poly_abs._pvs.shape
    residuals = pred_pvs - poly_abs._pvs
    unreduced_loss = scaled_pseudo_huber(delta, residuals)
    assert unreduced_loss.shape == poly_abs._pvs.shape
    if poly_abs._weights is None:
        return unreduced_loss.sum()
    else:
        assert unreduced_loss.shape == poly_abs._weights.shape
        return (poly_abs._weights * unreduced_loss).sum()


@partial(
    jit,
    static_argnames=[
        "poly_abs",
        "bv_sparse",
        "loss_delta",
        "reg_escape_weight",
        "reg_escape_delta",
        "reg_spread_weight",
        "matrix_to_mean",
        "coeff_positions",
    ],
)
def cost(
    params,
    poly_abs,
    bv_sparse,
    loss_delta,
    reg_escape_weight,
    reg_escape_delta,
    reg_spread_weight,
    matrix_to_mean,
    coeff_positions,
):
    """
    The cost is the loss with the regularization.
    """
    _, beta = a_beta_from_params(params, poly_abs)
    reg_escape = reg_escape_weight * scaled_pseudo_huber(reg_escape_delta, beta).sum()
    reg_spread = spread_penalty(
        beta, matrix_to_mean, coeff_positions, reg_spread_weight
    )
    return reg_escape + reg_spread + loss(params, poly_abs, bv_sparse, loss_delta)
