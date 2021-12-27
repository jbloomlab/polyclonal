"""
==========
functional
==========

Core functionality expressed as free functions.

n_epitopes = len(poly_abs.epitopes)
n_mutations = len(poly_abs.mutations)
a_beta_from_params(n_epitopes, n_mutations, params)

bmap


"""

from functools import partial
import numpy
import jax.numpy as jnp
from jax import jit


@partial(jit, static_argnames=["n_epitopes", "n_mutations"])
def a_beta_from_params(n_epitopes, n_mutations, params):
    """Vector of activities and MxE matrix of betas from params vector."""
    params_len = n_epitopes * (1 + n_mutations)
    if params.shape != (params_len,):
        raise ValueError(f"invalid {params.shape=}")
    a = params[: n_epitopes]
    beta = params[n_epitopes:].reshape(n_mutations, n_epitopes)
    assert a.shape == (n_epitopes,)
    assert beta.shape == (n_mutations, n_epitopes)
    assert (not numpy.isnan(a).any()) and (not numpy.isnan(beta).any())
    return (a, beta)


@partial(jit, static_argnames=["n_epitopes", "n_mutations", "n_variants"])
def compute_pv(n_epitopes, n_mutations, n_variants, params, bv_dense, cs):
    a, beta = a_beta_from_params(n_epitopes, n_mutations, params)
    phi_e_v = bv_dense @ beta - a
    assert phi_e_v.shape == (n_variants, n_epitopes)
    exp_minus_phi_e_v = jnp.exp(-phi_e_v)
    U_v_e_c = 1.0 / (1.0 + jnp.tensordot(exp_minus_phi_e_v, cs, axes=((), ())))
    assert U_v_e_c.shape == (n_variants, n_epitopes, len(cs))
    n_vc = n_variants * len(cs)
    U_vc_e = jnp.moveaxis(U_v_e_c, 1, 2).reshape(
                n_vc, n_epitopes, order='F')
    assert U_vc_e.shape == (n_vc, n_epitopes)
    p_vc = U_vc_e.prod(axis=1)
    assert p_vc.shape == (n_vc,)
    return p_vc
