"""
Tests for `loss.py`.
"""

import Bio.SeqIO
import pandas as pd
import pytest

import jax
import jax.numpy as jnp
from jax import jacrev
from jax.experimental import sparse
import jax.scipy.optimize

import dms_variants.simulate
import polyclonal
import polyclonal.loss as loss

# Make JAX use double precision numbers.
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def mini_activity_wt_df():
    return pd.read_csv("mini_activity_wt_df.csv")


@pytest.fixture
def mini_mut_escape_df():
    return pd.read_csv("mini_mut_escape_df.csv")


@pytest.fixture
def mini_data():
    return pd.read_csv("mini_escape_variants_exact.csv", na_filter=None).reset_index(
        drop=True
    )


@pytest.fixture
def mini_poly_abs_prefit(mini_data, mini_activity_wt_df, mini_mut_escape_df):
    return polyclonal.Polyclonal(
        data_to_fit=mini_data,
        activity_wt_df=mini_activity_wt_df,
        mut_escape_df=mini_mut_escape_df,
    )


@pytest.fixture
def exact_data():
    return (
        pd.read_csv("notebooks/RBD_variants_escape_exact.csv", na_filter=None)
        .query('library == "avg2muts"')
        .query("concentration in [0.25, 1, 4]")
        .reset_index(drop=True)
    )


@pytest.fixture
def exact_mut_escape_df():
    return pd.read_csv("exact_mut_escape_df.csv")


@pytest.fixture
def exact_activity_wt_df():
    return pd.read_csv("exact_activity_wt_df.csv")


@pytest.fixture
def poly_abs_prefit(exact_data, exact_activity_wt_df, exact_mut_escape_df):
    return polyclonal.Polyclonal(
        data_to_fit=exact_data,
        activity_wt_df=exact_activity_wt_df,
        mut_escape_df=exact_mut_escape_df,
    )


@pytest.fixture
def exact_bv_sparse(poly_abs_prefit):
    return loss.bv_sparse_of_bmap(poly_abs_prefit._binarymaps)


def test_compute_pv(mini_poly_abs_prefit):
    bv_sparse = loss.bv_sparse_of_bmap(mini_poly_abs_prefit._binarymaps)
    params = mini_poly_abs_prefit._params
    jax_pv = loss.full_pv(mini_poly_abs_prefit, bv_sparse, params)
    correct_pv, correct_pv_jac = mini_poly_abs_prefit._compute_pv(
        params,
        mini_poly_abs_prefit._binarymaps,
        cs=mini_poly_abs_prefit._cs,
        calc_grad=True,
    )
    assert jnp.allclose(jax_pv, jnp.array(correct_pv))
    jac_compute_pv = jacrev(loss.full_pv, argnums=2)
    jax_pv_jac = jac_compute_pv(mini_poly_abs_prefit, bv_sparse, params).transpose()
    assert jnp.allclose(jax_pv_jac, correct_pv_jac.todense())


def test_pseudo_huber():
    delta = 2.0
    r = [1.0, 2.0, 4.0, 8.0]
    h, hgrad = polyclonal.Polyclonal._scaled_pseudo_huber(delta, r, True)
    jax_h = loss.scaled_pseudo_huber(delta, jnp.array(r))
    assert jnp.allclose(h, jax_h)

    # Argnums specifies here that we want the gradient WRT the second argument.
    huberish_jac = jacrev(loss.scaled_pseudo_huber, argnums=1)
    jax_hgrad = jnp.diag(huberish_jac(delta, jnp.array(r)))
    assert jnp.allclose(hgrad, jax_hgrad)


def test_spread_penalty(poly_abs_prefit):
    (matrix_to_mean, coeff_positions) = loss.spread_matrices_of_polyclonal(
        poly_abs_prefit
    )
    n_epitopes = len(poly_abs_prefit.epitopes)
    n_mutations = len(poly_abs_prefit.mutations)
    _, beta = loss.a_beta_from_params(n_epitopes, n_mutations, poly_abs_prefit._params)
    jax_penalty = loss.spread_penalty(matrix_to_mean, coeff_positions, beta)
    correct_penalty, _ = poly_abs_prefit._reg_spread(poly_abs_prefit._params, 1.0)
    jax_penalty == pytest.approx(correct_penalty)


def test_unregularized_loss(poly_abs_prefit, exact_bv_sparse):
    delta = 0.1
    params = poly_abs_prefit._params
    jax_loss = loss.unregularized_loss(params, poly_abs_prefit, exact_bv_sparse, delta)
    prefit_loss, prefit_dloss = poly_abs_prefit._loss_dloss(params, delta)
    assert jax_loss == pytest.approx(prefit_loss)
    loss_grad = jax.grad(loss.unregularized_loss, 0)
    jax_loss_grad = loss_grad(params, poly_abs_prefit, exact_bv_sparse, delta)
    assert jnp.allclose(prefit_dloss, jax_loss_grad)


def test_loss(poly_abs_prefit, exact_bv_sparse):
    loss_delta = 0.15
    reg_escape_weight = 0.314
    reg_escape_delta = 0.29
    reg_spread_weight = 100.27
    params = poly_abs_prefit._params
    (matrix_to_mean, coeff_positions) = loss.spread_matrices_of_polyclonal(
        poly_abs_prefit
    )
    loss_args = [
        params,
        poly_abs_prefit,
        exact_bv_sparse,
        loss_delta,
        reg_escape_weight,
        reg_escape_delta,
        reg_spread_weight,
        matrix_to_mean,
        coeff_positions,
    ]
    jax_loss = loss.loss(*loss_args)
    fitloss, dfitloss = poly_abs_prefit._loss_dloss(params, loss_delta)
    regescape, dregescape = poly_abs_prefit._reg_escape(
        params, reg_escape_weight, reg_escape_delta
    )
    regspread, dregspread = poly_abs_prefit._reg_spread(params, reg_spread_weight)
    correct_loss = fitloss + regescape + regspread
    assert jax_loss == pytest.approx(correct_loss)
    correct_dloss = dfitloss + dregescape + dregspread
    loss_grad = jax.grad(loss.loss, 0)
    jax_loss_grad = loss_grad(*loss_args)
    diff = correct_dloss - jax_loss_grad
    print(jnp.abs(diff).max())
    assert jnp.allclose(correct_dloss, jax_loss_grad)