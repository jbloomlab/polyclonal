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

import dms_variants.simulate
import polyclonal
import polyclonal.loss as loss

# Make JAX use double precision numbers.
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def activity_wt_df():
    return pd.read_csv("mini_activity_wt_df.csv")


@pytest.fixture
def mut_escape_df():
    return pd.read_csv("mini_mut_escape_df.csv")


@pytest.fixture
def poly_abs(activity_wt_df, mut_escape_df):
    return polyclonal.Polyclonal(
        activity_wt_df=activity_wt_df, mut_escape_df=mut_escape_df
    )


@pytest.fixture
def geneseq():
    geneseq = str(Bio.SeqIO.read("notebooks/RBD_seq.fasta", "fasta").seq)
    return geneseq[:333]


@pytest.fixture
def variants_df(poly_abs, geneseq):
    allowed_aa_muts = poly_abs.mut_escape_df["mutation"].unique()
    variants = dms_variants.simulate.simulate_CodonVariantTable(
        geneseq=geneseq,
        bclen=16,
        library_specs={f"avg{m}muts": {"avgmuts": m, "nvariants": 4} for m in [1, 2]},
        allowed_aa_muts=[
            polyclonal.utils.shift_mut_site(m, -330) for m in allowed_aa_muts
        ],
    )

    return variants.barcode_variant_df[
        ["library", "barcode", "aa_substitutions", "n_aa_substitutions"]
    ].assign(
        aa_substitutions=lambda x: x["aa_substitutions"].apply(
            polyclonal.utils.shift_mut_site, shift=330
        )
    )


@pytest.fixture
def concentrations():
    return [0.25, 2]


@pytest.fixture
def variants_escape(poly_abs, variants_df, concentrations):
    variants_escape = poly_abs.prob_escape(
        variants_df=variants_df, concentrations=concentrations
    )
    variants_escape.rename(
        columns={"predicted_prob_escape": "prob_escape"}, inplace=True
    )
    return variants_escape


@pytest.fixture
def binarymaps_from_df_relevant_result(poly_abs, variants_escape):
    (one_binarymap, binarymaps, cs, pvs, _, variants_df) = poly_abs._binarymaps_from_df(
        variants_escape, get_pv=False, collapse_identical_variants=False
    )
    assert one_binarymap
    return (binarymaps, cs, variants_df)


@pytest.fixture
def bmap(binarymaps_from_df_relevant_result):
    return binarymaps_from_df_relevant_result[0]


@pytest.fixture
def cs(binarymaps_from_df_relevant_result):
    return binarymaps_from_df_relevant_result[1]


@pytest.fixture
def pvs(binarymaps_from_df_relevant_result):
    return binarymaps_from_df_relevant_result[2]


@pytest.fixture
def final_variants_df(binarymaps_from_df_relevant_result):
    return binarymaps_from_df_relevant_result[2]


@pytest.fixture
def n_epitopes(poly_abs):
    return len(poly_abs.epitopes)


@pytest.fixture
def n_mutations(poly_abs):
    return len(poly_abs.mutations)


@pytest.fixture
def bv_dense(bmap):
    return jnp.array(bmap.binary_variants.todense())


@pytest.fixture
def bv_sparse(bv_dense):
    return sparse.BCOO.fromdense(bv_dense)


@pytest.fixture
def params(poly_abs):
    return poly_abs._params


def test_poly_abs(poly_abs):
    assert len(poly_abs.epitopes) == 2
    assert len(poly_abs.sites) == 3


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


def test_compute_pv_2(poly_abs_prefit, exact_bv_sparse):
    params = poly_abs_prefit._params
    jax_pv = loss.full_pv(poly_abs_prefit, exact_bv_sparse, params)
    correct_pv, correct_pv_jac = poly_abs_prefit._compute_pv(
        params, poly_abs_prefit._binarymaps, cs=poly_abs_prefit._cs, calc_grad=True
    )
    assert jnp.allclose(jax_pv, jnp.array(correct_pv))
    # We can't do this for the big example because it takes too much memory.
    # jac_compute_pv = jacrev(loss.full_pv, argnums=2)
    # # TODO note transpose here.
    # jax_pv_jac = jac_compute_pv(poly_abs_prefit, exact_bv_sparse, params).transpose()
    # assert jnp.allclose(jax_pv_jac, correct_pv_jac.todense())


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


def test_compute_pv(poly_abs, n_epitopes, n_mutations, bmap, params, bv_sparse, cs):
    jax_pv = loss.compute_pv(
        n_epitopes, n_mutations, bmap.nvariants, params, bv_sparse, cs
    )
    correct_pv, correct_pv_jac = poly_abs._compute_pv(
        params, bmap, cs=cs, calc_grad=True
    )
    assert jnp.allclose(jax_pv, jnp.array(correct_pv))
    jac_compute_pv = jacrev(loss.compute_pv, argnums=3)
    # TODO note transpose here.
    jax_pv_jac = jac_compute_pv(
        n_epitopes, n_mutations, bmap.nvariants, params, bv_sparse, cs
    ).transpose()
    assert jnp.allclose(jax_pv_jac, correct_pv_jac.todense())


def test_loss(poly_abs_prefit, exact_bv_sparse):
    delta = 0.1
    params = poly_abs_prefit._params
    jax_loss = loss.loss(poly_abs_prefit, exact_bv_sparse, delta, params)
    prefit_loss, prefit_dloss = poly_abs_prefit._loss_dloss(params, delta)
    assert jax_loss == pytest.approx(prefit_loss)
    loss_grad = jax.grad(loss.loss, 3)
    jax_loss_grad = loss_grad(poly_abs_prefit, exact_bv_sparse, delta, params)
    assert jnp.allclose(prefit_dloss, jax_loss_grad)


def test_fit(poly_abs_prefit, exact_bv_sparse):
    optimize_result = jax.scipy.optimize.minimize(
        loss.loss,
        poly_abs_prefit._params,
        args=(poly_abs_prefit, exact_bv_sparse, 0.1),
        method="BFGS",
    )
    breakpoint()
