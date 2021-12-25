"""
==========
functional
==========

Core functionality expressed as free functions.

n_epitopes = len(poly_abs.epitopes)
n_mutations = len(poly_abs.mutations)
a_beta_from_params(n_epitopes, n_mutations, params)


"""

from functools import partial
import numpy

# @partial(jit, static_argnames=["n_epitopes", "n_mutations"])
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


