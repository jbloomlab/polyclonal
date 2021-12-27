"""
Tests for `loss.py`.
"""

import Bio.SeqIO
import pandas as pd
import pytest
from jax import jacrev
from jax.experimental import sparse
import jax.numpy as jnp

import dms_variants.simulate
import polyclonal
import polyclonal.loss as loss


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
    (one_binarymap, binarymaps, cs, _, _, variants_df) = poly_abs._binarymaps_from_df(
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
    correct_pv = jnp.array(poly_abs._compute_pv(params, bmap, cs=cs))
    assert jnp.allclose(jax_pv, correct_pv)
