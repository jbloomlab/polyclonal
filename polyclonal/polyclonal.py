"""
==========
polyclonal
==========

Defines :class:`Polyclonal` objects for handling antibody mixtures.

"""


import collections
import copy  # noqa: F401
import inspect
import itertools
import os
import sys
import time


import binarymap

import frozendict

import natsort

import numpy

import pandas as pd

import scipy.optimize
import scipy.special

import polyclonal
import polyclonal.alphabets
import polyclonal.pdb_utils
import polyclonal.plot
import polyclonal.utils


class PolyclonalFitError(Exception):
    """Error fitting in :meth:`Polyclonal.fit`."""

    pass


class PolyclonalHarmonizeError(Exception):
    """Error harmonizing epitopes in :meth:`Polyclonal.epitope_harmonized_model`."""

    pass


class Polyclonal:
    r"""Represent polyclonal antibody mixtures targeting multiple epitopes.

    Note
    ----
    At several concentrations :math:`c` of an antibody mixture, we measure
    :math:`p_v\left(c\right)`, the probability that variant :math:`v` is
    **not** bound (or neutralized). We assume antibodies act independently on
    one of :math:`E` epitopes, so the probability :math:`U_e\left(v, c\right)`
    that :math:`v` is unbound at concentration :math:`c` is related to the
    probability that epitope :math:`e` is unbound by

    .. math::
       :label: p_v

       p_v\left(c\right) = \prod_{e=1}^E U_e\left(v, c\right).

    We furthermore assume that :math:`U_e\left(v, c\right)` is related to the
    total binding activity :math:`\phi_e\left(v\right)` of antibodies targeting
    epitope :math:`e` on variant :math:`v` by

    .. math::
       :label: U_e

       U_e\left(v,c\right)=\frac{1}{1+c\exp\left(-\phi_e\left(v\right)\right)}

    where smaller (more negative) values of :math:`\phi_e\left(v\right)`
    correspond to higher overall binding activity against epitope :math:`e`
    variant :math:`v`.

    We define :math:`\phi_e\left(v\right)` in terms of the underlying
    quantities of biological interest as

    .. math::
       :label: phi_

       \phi_e\left(v\right) = -a_{\rm{wt}, e} +
                              \sum_{m=1}^M \beta_{m,e} b\left(v\right)_m,

    where :math:`a_{\rm{wt}, e}` is the activity of the serum against
    epitope :math:`e` for the "wildtype" (unmutated) protein (larger values
    indicate higher activity against this epitope), :math:`\beta_{m,e}`
    is the extent to which mutation :math:`m` (where :math:`1 \le m \le M`)
    escapes binding from antibodies targeting epitope :math:`e` (larger
    values indicate more escape by this mutation), and
    :math:`b\left(v\right)_m` is 1 if variant :math:`v` has mutation :math:`m`
    and 0 otherwise.

    Note
    ----
    You can initialize a :class:`Polyclonal` object in three ways:

    1. With known epitope activities :math:`a_{\rm{wt}, e}` and mutation-escape
       values :math:`\beta_{m,e}`, and no data. Use this approach if you
       already know these values and just want to visualize the polyclonal
       antibody mixture properties or predict escape of variants. To do this,
       initialize with ``activity_wt_df`` and ``mut_escape_df`` storing the
       known values, and ``data_to_fit=None``.

    2. With data to fit the epitope activities and mutation-escape values,
       and initial guesses for the epitope activities and mutation-escape
       values. To do this, initialize with ``data_to_fit`` holding the data,
       ``activity_wt_df`` holding initial guesses of activities, and
       ``mut_escape_df`` or ``site_escape_df`` holding initial guesses for
       mutation escapes (see also ``init_missing`` and
       ``data_mut_escape_overlap``). Then call :meth:`Polyclonal.fit`.

    3. With data to fit the epitope activities and mutation-escape values,
       but no initial guesses. To do this, initialize with ``data_to_fit``
       holding the data, ``activity_wt_df=None``, ``mut_escape_df=None``,
       and ``n_epitopes`` holding the number of epitopes. Then call
       :meth:`Polyclonal.fit`.

    Parameters
    ----------
    data_to_fit : pandas.DataFrame or None
        Should have columns named 'aa_substitutions', 'concentration', and
        'prob_escape'. The 'aa_substitutions' column defines each variant
        :math:`v` as a string of substitutions (e.g., 'M3A K5G'). The
        'prob_escape' column gives the :math:`p_v\left(c\right)` value for
        each variant at each concentration :math:`c`.
    activity_wt_df : pandas.DataFrame or None
        Should have columns named 'epitope' and 'activity', giving the names
        of the epitopes and the activity against epitope in the wildtype
        protein, :math:`a_{\rm{wt}, e}`.
    mut_escape_df : pandas.DataFrame or None
        Should have columns named 'mutation', 'epitope', and 'escape' that
        give the :math:`\beta_{m,e}` values (in the 'escape' column), with
        mutations written like "G7M".
    site_escape_df : pandas.DataFrame or None
        Use if you want to initialize all mutations at a given site to have
        the same :math:`\beta_{m,e}` values. In this case, columns should be
        'site', 'epitope', and 'escape'. This option is mutually exclusive
        with ``mut_escape_df``.
    n_epitopes : int or None
        If initializing with ``activity_wt_df=None``, specifies number
        of epitopes.
    collapse_identical_variants : {'mean', 'median', False}
        If identical variants in ``data_to_fit`` (same 'aa_substitutions'),
        collapse them and make weight proportional to number of collapsed
        variants? Collapse by taking mean or median of 'prob_escape', or
        (if `False`) do not collapse at all. Collapsing will make fitting faster,
        but *not* a good idea if you are doing bootstrapping.
    alphabet : array-like
        Allowed characters in mutation strings.
    sites : array-like or None
        By default, sites are assumed to be sequential integer values are and inferred
        from ``data_to_fit`` or ``mut_escape_df``. However, you can also have
        non-sequential integer sites, or sites with lower-case letter suffixes
        (eg, `214a`) if your protein is numbered against a reference that it has
        indels relative to. In that case, provide list of all expected in order
        here; we require that order to be natsorted.
    epitope_colors : array-like or dict
        Maps each epitope to the color used for plotting. Either a dict keyed
        by each epitope, or an array of colors that are sequentially assigned
        to the epitopes.
    init_missing : 'zero' or int
        How to initialize activities or mutation-escape values not specified in
        ``activity_wt_df`` or ``mut_escape_df`` / ``site_escape_df``. If
        'zero', set mutation-escapes to zero and activities uniformly spaced
        from 1 to 0. Otherwise draw uniformly from between 0 and 1 using
        specified random number seed.
    data_mut_escape_overlap : {'exact_match', 'fill_to_data', 'prune_to_data'}
        If ``data_to_fit`` and ``mut_escape_df`` (or ``site_escape_df``) both
        specified, what if they don't specify same mutations.
        If 'exact_match', raise error. If 'fill_to_data', then take
        sites / wildtypes / mutations from ``data_to_fit`` and fill init
        values from any not specified in ``mut_escape_df`` as indicated by
        ``init_missing``. If 'prune_to_data', remove any extra mutations
        from ``mut_escape_df`` that are not in ``data_to_fit``.

    Attributes
    ----------
    epitopes : tuple
        Names of all epitopes.
    mutations : tuple
        All mutations for which we have escape values.
    alphabet : tuple
        Allowed characters in mutation strings.
    sites : tuple
        List of all sites. These are the sites provided via the ``sites`` parameter,
        or inferred from ``data_to_fit`` or ``mut_escape_df`` if that isn't provided.
        If `sequential_integer_sites` is `False`, these are str, otherwise int.
    sequential_integer_sites : bool
        True if sites are sequential and integer, False otherwise.
    wts : dict
        Keyed by site, value is wildtype at that site.
    epitope_colors : dict
        Maps each epitope to its color.
    data_to_fit : pandas.DataFrame or None
        Data to fit as passed when initializing this :class:`Polyclonal` object.
        If using ``collapse_identical_variants``, then identical variants
        are collapsed on columns 'concentration', 'aa_substitutions',
        and 'prob_escape', and a column 'weight' is added to represent number
        of collapsed variants. Also, row-order may be changed.
    mutations_times_seen : frozendict.frozendict or None
        If `data_to_fit` is not `None`, keyed by all mutations with escape values
        and values are number of variants in which the mutation is seen. It is formally
        calculated as the number of variants with mutation across all concentrations
        divided by the number of concentrations, so can have non-integer values if
        there are variants only observed at some concentrations.

    Example
    -------
    Simple example with two epitopes (`e1` and `e2`) and a few mutations where
    we know the activities and mutation-level escape values ahead of time:

    >>> activity_wt_df = pd.DataFrame({'epitope':  ['e1', 'e2'],
    ...                                'activity': [ 2.0,  1.0]})
    >>> mut_escape_df = pd.DataFrame({
    ...   'mutation': ['M1C', 'M1C', 'G2A', 'G2A', 'A4K', 'A4K', 'A4L', 'A4L'],
    ...   'epitope':  [ 'e1',  'e2',  'e1',  'e2',  'e1',  'e2',  'e1',  'e2'],
    ...   'escape':   [  2.0,   0.0,   3.0,   0.0,  0.0,    2.5,   0.0,   1.5],
    ...   })
    >>> model = Polyclonal(
    ...     activity_wt_df=activity_wt_df,
    ...     mut_escape_df=mut_escape_df,
    ...     collapse_identical_variants="mean",
    ... )
    >>> model.epitopes
    ('e1', 'e2')
    >>> model.mutations
    ('M1C', 'G2A', 'A4K', 'A4L')
    >>> model.mutations_times_seen is None
    True
    >>> model.sites
    (1, 2, 4)
    >>> model.wts
    {1: 'M', 2: 'G', 4: 'A'}
    >>> model.activity_wt_df
      epitope  activity
    0      e1       2.0
    1      e2       1.0
    >>> model.mut_escape_df
      epitope  site wildtype mutant mutation  escape
    0      e1     1        M      C      M1C     2.0
    1      e1     2        G      A      G2A     3.0
    2      e1     4        A      K      A4K     0.0
    3      e1     4        A      L      A4L     0.0
    4      e2     1        M      C      M1C     0.0
    5      e2     2        G      A      G2A     0.0
    6      e2     4        A      K      A4K     2.5
    7      e2     4        A      L      A4L     1.5

    We can also summarize the mutation-level escape at the site level:

    >>> pd.set_option("display.max_columns", None)
    >>> pd.set_option("display.width", 89)
    >>> model.mut_escape_site_summary_df()
      epitope  site wildtype  mean  total positive  max  min  total negative  n mutations
    0      e1     1        M   2.0             2.0  2.0  2.0             0.0            1
    1      e1     2        G   3.0             3.0  3.0  3.0             0.0            1
    2      e1     4        A   0.0             0.0  0.0  0.0             0.0            2
    3      e2     1        M   0.0             0.0  0.0  0.0             0.0            1
    4      e2     2        G   0.0             0.0  0.0  0.0             0.0            1
    5      e2     4        A   2.0             4.0  2.5  1.5             0.0            2

    Note that we can **not** initialize a :class:`Polyclonal` object if we are
    missing escape estimates for any mutations for any epitopes:

    >>> Polyclonal(activity_wt_df=activity_wt_df,
    ...            mut_escape_df=mut_escape_df.head(n=5))
    Traceback (most recent call last):
      ...
    ValueError: invalid set of mutations for epitope='e2'

    Now make a data frame with some variants:

    >>> variants_df = pd.DataFrame.from_records(
    ...         [('AA', ''),
    ...          ('AC', 'M1C'),
    ...          ('AG', 'G2A'),
    ...          ('AT', 'A4K'),
    ...          ('TA', 'A4L'),
    ...          ('CA', 'M1C G2A'),
    ...          ('CG', 'M1C A4K'),
    ...          ('CC', 'G2A A4K'),
    ...          ('TC', 'G2A A4L'),
    ...          ('CT', 'M1C G2A A4K'),
    ...          ('TG', 'M1C G2A A4L'),
    ...          ('GA', 'M1C'),
    ...          ],
    ...         columns=['barcode', 'aa_substitutions'])

    Get the escape probabilities predicted on these variants from
    the values in the :class:`Polyclonal` object:

    >>> escape_probs = model.prob_escape(variants_df=variants_df,
    ...                                  concentrations=[1.0, 2.0, 4.0])
    >>> escape_probs.round(3)
       barcode aa_substitutions  concentration  predicted_prob_escape
    0       AA                             1.0                  0.032
    1       AT              A4K            1.0                  0.097
    2       TA              A4L            1.0                  0.074
    3       AG              G2A            1.0                  0.197
    4       CC          G2A A4K            1.0                  0.598
    5       TC          G2A A4L            1.0                  0.455
    6       AC              M1C            1.0                  0.134
    7       GA              M1C            1.0                  0.134
    8       CG          M1C A4K            1.0                  0.409
    9       CA          M1C G2A            1.0                  0.256
    10      CT      M1C G2A A4K            1.0                  0.779
    11      TG      M1C G2A A4L            1.0                  0.593
    12      AA                             2.0                  0.010
    13      AT              A4K            2.0                  0.044
    14      TA              A4L            2.0                  0.029
    15      AG              G2A            2.0                  0.090
    16      CC          G2A A4K            2.0                  0.398
    17      TC          G2A A4L            2.0                  0.260
    18      AC              M1C            2.0                  0.052
    19      GA              M1C            2.0                  0.052
    20      CG          M1C A4K            2.0                  0.230
    21      CA          M1C G2A            2.0                  0.141
    22      CT      M1C G2A A4K            2.0                  0.629
    23      TG      M1C G2A A4L            2.0                  0.411
    24      AA                             4.0                  0.003
    25      AT              A4K            4.0                  0.017
    26      TA              A4L            4.0                  0.010
    27      AG              G2A            4.0                  0.034
    28      CC          G2A A4K            4.0                  0.214
    29      TC          G2A A4L            4.0                  0.118
    30      AC              M1C            4.0                  0.017
    31      GA              M1C            4.0                  0.017
    32      CG          M1C A4K            4.0                  0.106
    33      CA          M1C G2A            4.0                  0.070
    34      CT      M1C G2A A4K            4.0                  0.441
    35      TG      M1C G2A A4L            4.0                  0.243

    We can also get predicted escape probabilities by including concentrations
    in the data frame passed to :meth:`Polyclonal.prob_escape`:

    >>> model.prob_escape(
    ...         variants_df=pd.concat([variants_df.assign(concentration=c)
    ...                                for c in [1.0, 2.0, 4.0]])
    ...         ).equals(escape_probs)
    True

    We can also compute the IC50s:

    >>> model.icXX(variants_df).round(3)
       barcode aa_substitutions   IC50
    0       AA                   0.085
    1       AC              M1C  0.230
    2       GA              M1C  0.230
    3       AG              G2A  0.296
    4       AT              A4K  0.128
    5       TA              A4L  0.117
    6       CA          M1C G2A  0.355
    7       CG          M1C A4K  0.722
    8       CC          G2A A4K  1.414
    9       TC          G2A A4L  0.858
    10      CT      M1C G2A A4K  3.237
    11      TG      M1C G2A A4L  1.430

    Or the IC90s:

    >>> model.icXX(variants_df, x=0.9, col='IC90').round(3)
       barcode aa_substitutions    IC90
    0       AA                    0.464
    1       AC              M1C   1.260
    2       GA              M1C   1.260
    3       AG              G2A   1.831
    4       AT              A4K   0.976
    5       TA              A4L   0.782
    6       CA          M1C G2A   2.853
    7       CG          M1C A4K   4.176
    8       CC          G2A A4K   7.473
    9       TC          G2A A4L   4.532
    10      CT      M1C G2A A4K  18.717
    11      TG      M1C G2A A4L   9.532

    Example
    -------
    Initialize with ``escape_probs`` created above as data to fit. In order
    to do this, we need to change the name of the column with the
    predicted escape probs to just be escape probs as we are now assuming
    these are the real values:

    >>> data_to_fit = (
    ...         escape_probs
    ...         .rename(columns={'predicted_prob_escape': 'prob_escape'})
    ...         )

    >>> model_data = Polyclonal(
    ...     data_to_fit=data_to_fit,
    ...     n_epitopes=2,
    ...     collapse_identical_variants="mean",
    ... )

    The mutations are those in ``data_to_fit``:

    >>> model_data.mutations
    ('M1C', 'G2A', 'A4K', 'A4L')
    >>> dict(model_data.mutations_times_seen)
    {'G2A': 6, 'M1C': 6, 'A4K': 4, 'A4L': 3}

    The activities are evenly spaced from 1 to 0, while the mutation escapes
    are all initialized to zero:

    >>> model_data.activity_wt_df
      epitope  activity
    0       1       1.0
    1       2       0.0
    >>> model_data.mut_escape_df
      epitope  site wildtype mutant mutation  escape  times_seen
    0       1     1        M      C      M1C     0.0           6
    1       1     2        G      A      G2A     0.0           6
    2       1     4        A      K      A4K     0.0           4
    3       1     4        A      L      A4L     0.0           3
    4       2     1        M      C      M1C     0.0           6
    5       2     2        G      A      G2A     0.0           6
    6       2     4        A      K      A4K     0.0           4
    7       2     4        A      L      A4L     0.0           3

    You can initialize to random numbers by setting ``init_missing`` to seed
    (in this example we also don't include all variants for one concentration):

    >>> model_data2 = Polyclonal(
    ...     data_to_fit=data_to_fit.head(30),
    ...     n_epitopes=2,
    ...     init_missing=1,
    ...     collapse_identical_variants="mean",
    ... )
    >>> model_data2.activity_wt_df.round(3)
      epitope  activity
    0       1     0.417
    1       2     0.720

    You can set some or all mutation escapes to initial values:

    >>> model_data3 = Polyclonal(
    ...     data_to_fit=data_to_fit,
    ...     activity_wt_df=activity_wt_df,
    ...     mut_escape_df=pd.DataFrame({'epitope': ['e1'],
    ...                                 'mutation': ['M1C'],
    ...                                 'escape': [4]}),
    ...     data_mut_escape_overlap='fill_to_data',
    ...     collapse_identical_variants="mean",
    ... )
    >>> model_data3.mut_escape_df
      epitope  site wildtype mutant mutation  escape  times_seen
    0      e1     1        M      C      M1C     4.0           6
    1      e1     2        G      A      G2A     0.0           6
    2      e1     4        A      K      A4K     0.0           4
    3      e1     4        A      L      A4L     0.0           3
    4      e2     1        M      C      M1C     0.0           6
    5      e2     2        G      A      G2A     0.0           6
    6      e2     4        A      K      A4K     0.0           4
    7      e2     4        A      L      A4L     0.0           3

    You can initialize **sites** to escape values via ``site_activity_df``:

    >>> model_data4 = Polyclonal(
    ...     data_to_fit=data_to_fit,
    ...     activity_wt_df=activity_wt_df,
    ...     site_escape_df=pd.DataFrame.from_records(
    ...         [('e1', 1, 1.0), ('e1', 4, 0.0),
    ...          ('e2', 1, 0.0), ('e2', 4, 2.0)],
    ...         columns=['epitope', 'site', 'escape'],
    ...     ),
    ...     data_mut_escape_overlap='fill_to_data',
    ...     collapse_identical_variants="mean",
    ... )
    >>> model_data4.mut_escape_df
      epitope  site wildtype mutant mutation  escape  times_seen
    0      e1     1        M      C      M1C     1.0           6
    1      e1     2        G      A      G2A     0.0           6
    2      e1     4        A      K      A4K     0.0           4
    3      e1     4        A      L      A4L     0.0           3
    4      e2     1        M      C      M1C     0.0           6
    5      e2     2        G      A      G2A     0.0           6
    6      e2     4        A      K      A4K     2.0           4
    7      e2     4        A      L      A4L     2.0           3

    Fit the data using :meth:`Polyclonal.fit`, and make sure the new
    predicted escape probabilities are close to the real ones being fit.
    Reduce weight on regularization since there is so little data in this
    toy example:

    >>> for m in [model_data, model_data2, model_data3, model_data4]:
    ...     opt_res = m.fit(
    ...         reg_escape_weight=0.001,
    ...         reg_spread_weight=0.001,
    ...         reg_activity_weight=0.0001,
    ...     )
    ...     pred_df = m.prob_escape(variants_df=data_to_fit)
    ...     if not numpy.allclose(pred_df['prob_escape'],
    ...                           pred_df['predicted_prob_escape'],
    ...                           atol=0.01):
    ...          raise ValueError(f"wrong predictions\n{pred_df}")
    ...     if not numpy.allclose(
    ...              activity_wt_df['activity'].sort_values(),
    ...              m.activity_wt_df['activity'].sort_values(),
    ...              atol=0.1,
    ...              ):
    ...          raise ValueError(f"wrong activities\n{m.activity_wt_df}")
    ...     if not numpy.allclose(
    ...              mut_escape_df['escape'].sort_values(),
    ...              m.mut_escape_df['escape'].sort_values(),
    ...              atol=0.05,
    ...              ):
    ...          raise ValueError(f"wrong escapes\n{m.mut_escape_df}")

    >>> model_data.mut_escape_site_summary_df().round(1)
      epitope  site wildtype  mean  total positive  max  min  total negative  n mutations
    0       1     1        M   0.0             0.0  0.0  0.0             0.0            1
    1       1     2        G   0.0             0.0  0.0  0.0             0.0            1
    2       1     4        A   2.0             4.0  2.5  1.5             0.0            2
    3       2     1        M   2.0             2.0  2.0  2.0             0.0            1
    4       2     2        G   3.0             3.0  3.0  3.0             0.0            1
    5       2     4        A   0.0             0.0  0.0  0.0             0.0            2
    >>> model_data.mut_escape_site_summary_df(min_times_seen=4).round(1)
      epitope  site wildtype  mean  total positive  max  min  total negative  n mutations
    0       1     1        M   0.0             0.0  0.0  0.0             0.0            1
    1       1     2        G   0.0             0.0  0.0  0.0             0.0            1
    2       1     4        A   2.5             2.5  2.5  2.5             0.0            1
    3       2     1        M   2.0             2.0  2.0  2.0             0.0            1
    4       2     2        G   3.0             3.0  3.0  3.0             0.0            1
    5       2     4        A   0.0             0.0  0.0  0.0             0.0            1

    You can also exclude mutations to specific characters (typically you would want to
    do this for stop codons and/or gaps):

    >>> model_data.mut_escape_site_summary_df(exclude_chars={"C", "K"}).round(1)
      epitope  site wildtype  mean  total positive  max  min  total negative  n mutations
    0       1     2        G   0.0             0.0  0.0  0.0             0.0            1
    1       1     4        A   1.5             1.5  1.5  1.5             0.0            1
    2       2     2        G   3.0             3.0  3.0  3.0             0.0            1
    3       2     4        A   0.0             0.0  0.0  0.0             0.0            1

    Example
    -------
    You can convert a :class:`Polyclonal` model into a site-level model via
    the transformation of :func:`polyclonal.utils.site_level_variants`. The
    site-level model is another :class:`Polyclonal` model that just keeps
    track of whether or not sites are mutated using a 2-letter wildtype/mutant
    alphabet, and is generated using :meth:`Polyclonal.site_level_model`:

    >>> model_site = model_data4.site_level_model()
    >>> model_site.alphabet
    ('w', 'm')
    >>> (model_site.mut_escape_df
    ...  .assign(escape=lambda x: x['escape'].abs()).round(1))
      epitope  site wildtype mutant mutation  escape  times_seen
    0      e1     1        w      m      w1m     2.0           5
    1      e1     2        w      m      w2m     3.0           6
    2      e1     4        w      m      w4m     0.0           7
    3      e2     1        w      m      w1m     0.0           5
    4      e2     2        w      m      w2m     0.0           6
    5      e2     4        w      m      w4m     2.0           7
    >>> model_site.data_to_fit.head(n=5).round(3)
       concentration aa_substitutions  weight  prob_escape
    0            1.0                        1        0.032
    1            1.0              w1m       1        0.134
    2            1.0          w1m w2m       1        0.256
    3            1.0      w1m w2m w4m       2        0.686
    4            1.0          w1m w4m       1        0.409

    Example
    -------
    Epitope assignments are arbitrary, so you can calculate correlations
    among their mutation-escape values and create harmonized models that use
    the same label to refer to epitopes with similar mutation-escape values.

    First, correlations of a model to itself:

    >>> ref_model = copy.deepcopy(model)
    >>> model.mut_escape_corr(ref_model).round(3)
      ref_epitope self_epitope  correlation
    0          e1           e1        1.000
    1          e1           e2       -0.907
    2          e2           e1       -0.907
    3          e2           e2        1.000

    Make another model with epitope assignments inverted and a few mutations missing:

    >>> inverted_model = Polyclonal(
    ...     activity_wt_df=(
    ...         activity_wt_df
    ...         .assign(epitope=lambda x: x["epitope"].map({"e1": "e2", "e2": "e1"}))
    ...     ),
    ...     mut_escape_df=(
    ...         mut_escape_df
    ...         .query("mutation != 'A4K'")
    ...         .assign(epitope=lambda x: x["epitope"].map({"e1": "e2", "e2": "e1"}))
    ...     ),
    ... )
    >>> inverted_model.mut_escape_corr(ref_model).round(3)
      ref_epitope self_epitope  correlation
    0          e1           e1       -0.945
    1          e1           e2        1.000
    2          e2           e1        1.000
    3          e2           e2       -0.945

    Now actually get epitope-harmonized models:

    >>> model_harmonized, harmonize_df = model.epitope_harmonized_model(ref_model)
    >>> harmonize_df
      self_initial_epitope self_harmonized_epitope ref_epitope  correlation
    0                   e1                      e1          e1          1.0
    1                   e2                      e2          e2          1.0
    >>> assert model.mut_escape_df.equals(model_harmonized.mut_escape_df)

    >>> inverted_harmonized, harmonize_df = inverted_model.epitope_harmonized_model(
    ...     ref_model
    ... )
    >>> harmonize_df
      self_initial_epitope self_harmonized_epitope ref_epitope  correlation
    0                   e1                      e2          e2          1.0
    1                   e2                      e1          e1          1.0
    >>> inverted_harmonized.mut_escape_df
      epitope  site wildtype mutant mutation  escape
    0      e1     1        M      C      M1C     2.0
    1      e1     2        G      A      G2A     3.0
    2      e1     4        A      L      A4L     0.0
    3      e2     1        M      C      M1C     0.0
    4      e2     2        G      A      G2A     0.0
    5      e2     4        A      L      A4L     1.5

    Example
    -------
    Filter variants by how often they are seen in data:

    >>> model_data.filter_variants_by_seen_muts(variants_df)
    ... # doctest: +NORMALIZE_WHITESPACE
       barcode aa_substitutions
    0       AA
    1       AC              M1C
    2       AG              G2A
    3       AT              A4K
    4       TA              A4L
    5       CA          M1C G2A
    6       CG          M1C A4K
    7       CC          G2A A4K
    8       TC          G2A A4L
    9       CT      M1C G2A A4K
    10      TG      M1C G2A A4L
    11      GA              M1C

    >>> model_data.filter_variants_by_seen_muts(variants_df, min_times_seen=5)
    ... # doctest: +NORMALIZE_WHITESPACE
      barcode aa_substitutions
    0      AA
    1      AC              M1C
    2      AG              G2A
    3      CA          M1C G2A
    4      GA              M1C

    >>> model_data.filter_variants_by_seen_muts(variants_df, min_times_seen=4)
    ... # doctest: +NORMALIZE_WHITESPACE
      barcode aa_substitutions
    0      AA
    1      AC              M1C
    2      AG              G2A
    3      AT              A4K
    4      CA          M1C G2A
    5      CG          M1C A4K
    6      CC          G2A A4K
    7      CT      M1C G2A A4K
    8      GA              M1C


    """

    def __init__(
        self,
        *,
        activity_wt_df=None,
        mut_escape_df=None,
        data_to_fit=None,
        site_escape_df=None,
        n_epitopes=None,
        collapse_identical_variants=False,
        alphabet=polyclonal.AAS,
        sites=None,
        epitope_colors=polyclonal.plot.DEFAULT_POSITIVE_COLORS,
        init_missing="zero",
        data_mut_escape_overlap="exact_match",
    ):
        """See main class docstring."""
        if isinstance(init_missing, int):
            numpy.random.seed(init_missing)
        elif init_missing != "zero":
            raise ValueError(f"invalid {init_missing=}")

        if sites is not None:
            sites = tuple(sites)
            if sites != tuple(natsort.natsorted(sites, alg=natsort.ns.SIGNED)):
                raise ValueError("`sites` not natsorted")
            if any(type(r) != int for r in sites) or sites != tuple(
                range(sites[0], sites[-1] + 1)
            ):
                self.sequential_integer_sites = False
                self.sites = tuple(map(str, sites))
            else:
                self.sequential_integer_sites = True
                self.sites = sites
        else:
            self.sites = None
            self.sequential_integer_sites = True

        if len(set(alphabet)) != len(alphabet):
            raise ValueError("duplicate letters in `alphabet`")
        self.alphabet = tuple(alphabet)
        self._mutparser = polyclonal.utils.MutationParser(
            alphabet,
            letter_suffixed_sites=not self.sequential_integer_sites,
        )

        # get any epitope labels as str, not int
        if activity_wt_df is not None:
            activity_wt_df = activity_wt_df.assign(
                epitope=lambda x: x["epitope"].astype(str)
            )
        if site_escape_df is not None:
            site_escape_df = site_escape_df.assign(
                epitope=lambda x: x["epitope"].astype(str)
            )
        if mut_escape_df is not None:
            mut_escape_df = mut_escape_df.assign(
                epitope=lambda x: x["epitope"].astype(str)
            )

        if site_escape_df is not None:
            if mut_escape_df is not None:
                raise ValueError(
                    "cannot set both `site_escape_df` and " "`mut_escape_df`"
                )
            if activity_wt_df is None:
                raise ValueError(
                    "cannot set `site_escape_df` without " "setting `activity_wt_df`"
                )
            if data_to_fit is None:
                raise ValueError(
                    "cannot set `site_escape_df` without " "setting `data_to_fit`"
                )
        if (activity_wt_df is not None) and (
            (mut_escape_df is not None) or (site_escape_df is not None)
        ):
            if n_epitopes is not None:
                raise ValueError("specify `activity_wt_df` or `n_epitopes`")

            if pd.isnull(activity_wt_df["epitope"]).any():
                raise ValueError("epitope name cannot be null")
            self.epitopes = tuple(activity_wt_df["epitope"].unique())
            if len(self.epitopes) != len(activity_wt_df):
                raise ValueError("duplicate epitopes in `activity_wt_df`")

        elif (activity_wt_df is None) and (mut_escape_df is None):
            if not (isinstance(n_epitopes, int) and n_epitopes > 0):
                raise ValueError(
                    "`n_epitopes` must be int > 1 if no " "`activity_wt_df`"
                )
            self.epitopes = tuple(f"{i + 1}" for i in range(n_epitopes))

            # initialize activities
            activity_wt_df = pd.DataFrame(
                {
                    "epitope": self.epitopes,
                    "activity": (
                        numpy.linspace(1, 0, len(self.epitopes))
                        if init_missing == "zero"
                        else numpy.random.rand(len(self.epitopes))
                    ),
                }
            )

            if data_to_fit is None:
                raise ValueError(
                    "specify `data_to_fit` if `activity_wt_df` "
                    "and `mut_escape_df` are `None`"
                )

        else:
            raise ValueError(
                "initialize both or neither `activity_wt_df` "
                "and `mut_escape_df` or `site_escape_df`"
            )
        if isinstance(epitope_colors, dict):
            self.epitope_colors = {e: epitope_colors[e] for e in self.epitopes}
        elif len(epitope_colors) < len(self.epitopes):
            raise ValueError("not enough `epitope_colors`")
        else:
            self.epitope_colors = dict(zip(self.epitopes, epitope_colors))

        def _init_mut_escape_df(mutations):
            # initialize mutation escape values
            if init_missing == "zero":
                init = 0.0
            else:
                init = numpy.random.rand(len(self.epitopes) * len(mutations))
            return pd.DataFrame(
                {
                    "epitope": list(self.epitopes) * len(mutations),
                    "mutation": [m for m in mutations for _ in self.epitopes],
                    "escape": init,
                }
            )

        # get wildtype, sites, and mutations
        if data_to_fit is not None:
            wts2, sites2, muts2 = self._muts_from_data_to_fit(data_to_fit)
            if (self.sites is not None) and not set(sites2).issubset(self.sites):
                raise ValueError("sites in `data_to_fit` not all in `sites`")
            times_seen = (
                data_to_fit["aa_substitutions"]
                .str.split()
                .explode()
                .dropna()
                .value_counts()
                .sort_values(ascending=False)
                / data_to_fit["concentration"].nunique()
            )
            if (times_seen == times_seen.astype(int)).all():
                times_seen = times_seen.astype(int)
            self.mutations_times_seen = frozendict.frozendict(times_seen)
        else:
            self.mutations_times_seen = None
        if site_escape_df is not None:
            # construct mut_escape_df from site_escape_df and mutations
            # from data_to_fit
            req_cols = {"epitope", "site", "escape"}
            if not req_cols.issubset(site_escape_df.columns):
                raise ValueError(f"`site_escape_df` lacks columns {req_cols}")
            assert (data_to_fit is not None) and (mut_escape_df is None)
            if not self.sequential_integer_sites:
                site_escape_df = site_escape_df.assign(
                    site=lambda x: x["site"].astype(str)
                )
            if not set(site_escape_df["epitope"]).issubset(self.epitopes):
                raise ValueError("`site_escape_df` has unrecognized epitopes")
            if not set(site_escape_df["site"]).issubset(sites2):
                raise ValueError("site_escape_df has sites not in data_to_fit")
            if len(site_escape_df) != len(
                site_escape_df[["site", "epitope"]].drop_duplicates()
            ):
                raise ValueError(
                    "`site_escape_df` rows do not each represent "
                    f"unique epitope / site:\n{site_escape_df}"
                )
            mut_records = []
            for epitope in self.epitopes:
                site_escape = (
                    site_escape_df.query("epitope == @epitope")
                    .set_index("site")["escape"]
                    .to_dict()
                )
                for mut in muts2:
                    (_, site, _) = self._mutparser.parse_mut(mut)
                    if site in site_escape:
                        mut_records.append((epitope, mut, site_escape[site]))
            mut_escape_df = pd.DataFrame.from_records(
                mut_records, columns=["epitope", "mutation", "escape"]
            )

        if mut_escape_df is not None:
            wts, sites, muts = self._muts_from_mut_escape_df(mut_escape_df)
            if (self.sites is not None) and not set(sites).issubset(self.sites):
                raise ValueError("`mut_escape_df` has sites not in `sites`")
        if mut_escape_df is data_to_fit is None:
            raise ValueError("initialize `mut_escape_df` or `data_to_fit`")
        elif mut_escape_df is None:
            self.wts, self.mutations = wts2, muts2
            if self.sites is None:
                self.sites = sites2
            mut_escape_df = _init_mut_escape_df(self.mutations)
        elif data_to_fit is None:
            self.wts, self.mutations = wts, muts
            if self.sites is None:
                self.sites = sites
        else:
            if data_mut_escape_overlap == "exact_match":
                if sites == sites2 and wts == wts2 and muts == muts2:
                    self.wts, self.mutations = wts, muts
                    if self.sites is None:
                        self.sites = sites
                else:
                    raise ValueError(
                        "`data_to_fit` and `mut_escape_df` give different mutations. "
                        "Fix or set data_mut_escape_overlap='fill_to_data'"
                    )
            elif data_mut_escape_overlap == "fill_to_data":
                # sites are in mut_escape_df, sites2 in data_to_fit
                if set(sites) <= set(sites2):
                    if self.sites is None:
                        self.sites = sites2
                    elif not set(self.sites).issuperset(sites2):
                        raise ValueError("sites in data_to_fit not subset of sites")
                else:
                    raise ValueError(
                        "`mut_escape_df` has more sites than `data_to_fit`"
                    )
                if wts.items() <= wts2.items():
                    self.wts = wts2
                else:
                    raise ValueError("`mut_escape_df` has wts not in `data_to_fit`")
                if set(muts) <= set(muts2):
                    self.mutations = muts2
                else:
                    raise ValueError(
                        "`mut_escape_df` has mutations not in `data_to_fit`"
                    )
                # take values from `mut_escape_df` and fill missing
                mut_escape_df = (
                    mut_escape_df.set_index(["epitope", "mutation"])["escape"]
                    .combine_first(
                        _init_mut_escape_df(self.mutations).set_index(
                            ["epitope", "mutation"]
                        )["escape"]
                    )
                    .reset_index()
                )
            elif data_mut_escape_overlap == "prune_to_data":
                # sites are in mut_escape_df, sites2 in data_to_fit
                if set(sites) >= set(sites2):
                    if self.sites is None:
                        self.sites = sites2
                    elif not set(self.sites).issuperset(sites2):
                        raise ValueError("sites in data_to_fit not subset of sites")
                else:
                    raise ValueError(
                        "`mut_escape_df` has fewer sites than `data_to_fit`"
                    )
                if wts.items() >= wts2.items():
                    self.wts = wts2
                else:
                    raise ValueError("`mut_escape_df` fewer wts than `data_to_fit`")
                if set(muts) >= set(muts2):
                    self.mutations = muts2
                else:
                    raise ValueError(
                        "`mut_escape_df` has fewer mutations than `data_to_fit`"
                    )
                mut_escape_df = mut_escape_df.query("mutation in @self.mutations")
                assert set(mut_escape_df["mutation"]) == set(self.mutations)
            else:
                raise ValueError(f"invalid {data_mut_escape_overlap=}")

        if set(mut_escape_df["epitope"]) != set(self.epitopes):
            raise ValueError(
                "`mut_escape_df` does not have same epitopes as " "`activity_wt_df`"
            )
        for epitope, df in mut_escape_df.groupby("epitope"):
            if sorted(df["mutation"]) != sorted(self.mutations):
                raise ValueError(f"invalid set of mutations for {epitope=}")

        # set internal params with activities and escapes
        self._params = self._params_from_dfs(activity_wt_df, mut_escape_df)

        if data_to_fit is not None:
            (
                self._one_binarymap,
                self._binarymaps,
                self._cs,
                self._pvs,
                self._weights,
                self.data_to_fit,
            ) = self._binarymaps_from_df(data_to_fit, True, collapse_identical_variants)
            assert len(self._pvs) == len(self.data_to_fit)
            # for each site get mask of indices in the binary map
            # that correspond to that site
            if self._one_binarymap:
                binary_sites = self._binarymaps.binary_sites
            else:
                binary_sites = self._binarymaps[0].binary_sites
                assert all(
                    (binary_sites == bmap.binary_sites).all()
                    for bmap in self._binarymaps
                )
            self._binary_sites = {
                site: numpy.where(binary_sites == site)
                for site in numpy.unique(binary_sites)
            }
        else:
            self.data_to_fit = None

    def _binarymaps_from_df(
        self,
        df,
        get_pv,
        collapse_identical_variants,
    ):
        """Get variants and and other information from data frame.

        Get `(one_binarymap, binarymaps, cs, pvs, weights, sorted_df)`. If
        `get_pv=False` then `pvs` is `None`. If `collapse_identical_variants`
        is `False` then `weights` is `None`. If same variants for all
        concentrations, `binarymaps` is a BinaryMap and `one_binarymap` is
        `True`. Otherwise, `binarymaps` lists BinaryMap for each concentration.
        `sorted_df` is version of `df` with variants/concentrations in same
        order as `binarymaps`, while `pvs` is 1D array with the prob escapes
        for these variants. We handle separately cases when BinaryMap same or
        different for concentrations as more efficient if all concentrations
        have same BinaryMap.

        """
        cols = ["concentration", "aa_substitutions"]
        if "weight" in df.columns:
            cols.append(
                "weight"
            )  # will be overwritten if `collapse_identical_variants`
        if get_pv:
            cols.append("prob_escape")
        if not df[cols].notnull().all().all():
            raise ValueError(f"null entries in data frame of variants:\n{df[cols]}")
        if collapse_identical_variants:
            agg_dict = {"weight": "sum"}
            if get_pv:
                agg_dict["prob_escape"] = collapse_identical_variants
            df = (
                df[cols]
                .assign(weight=1)
                .groupby(["concentration", "aa_substitutions"], as_index=False)
                .aggregate(agg_dict)
            )
        sorted_df = df.sort_values(["concentration", "aa_substitutions"]).reset_index(
            drop=True
        )
        cs = sorted_df["concentration"].astype(float).sort_values().unique()
        if not (cs > 0).all():
            raise ValueError("concentrations must be > 0")
        binarymaps = []
        pvs = [] if get_pv else None
        weights = [] if collapse_identical_variants else None
        one_binarymap = True
        for i, (c, i_df) in enumerate(sorted_df.groupby("concentration", sort=False)):
            assert c == cs[i]
            i_variants = i_df["aa_substitutions"].reset_index(drop=True)
            if i == 0:
                first_variants = i_variants
            elif one_binarymap:
                one_binarymap = first_variants.equals(i_variants)
            binarymaps.append(self._get_binarymap(i_df))
            if get_pv:
                pvs.append(i_df["prob_escape"].to_numpy(dtype=float))
            if collapse_identical_variants:
                weights.append(i_df["weight"].to_numpy(dtype=int))
        if one_binarymap:
            binarymaps = binarymaps[0]
        if get_pv:
            pvs = numpy.concatenate(pvs)
            assert len(pvs) == len(sorted_df)
            if (pvs < 0).any() or (pvs > 1).any():
                raise ValueError("`prob_escape` must be between 0 and 1")
        if collapse_identical_variants:
            weights = numpy.concatenate(weights)
            assert len(weights) == len(sorted_df)
            assert (weights >= 1).all()

        return (one_binarymap, binarymaps, cs, pvs, weights, sorted_df)

    def _params_from_dfs(self, activity_wt_df, mut_escape_df):
        """Params vector from data frames of activities and escapes."""
        # first E entries are activities
        assert len(activity_wt_df) == len(self.epitopes)
        assert len(self.epitopes) == activity_wt_df["epitope"].nunique()
        assert set(self.epitopes) == set(activity_wt_df["epitope"])
        params = (
            activity_wt_df.assign(
                epitope=lambda x: pd.Categorical(
                    x["epitope"], self.epitopes, ordered=True
                )
            )
            .sort_values("epitope")["activity"]
            .tolist()
        )

        # Remaining MxE entries are beta values
        assert len(mut_escape_df) == len(self.epitopes) * len(self.mutations)
        assert len(mut_escape_df) == len(mut_escape_df.groupby(["mutation", "epitope"]))
        assert set(self.epitopes) == set(mut_escape_df["epitope"])
        assert set(self.mutations) == set(mut_escape_df["mutation"])
        params.extend(
            mut_escape_df.assign(
                epitope=lambda x: pd.Categorical(
                    x["epitope"], self.epitopes, ordered=True
                ),
                mutation=lambda x: pd.Categorical(
                    x["mutation"], self.mutations, ordered=True
                ),
            )
            .sort_values(["mutation", "epitope"])["escape"]
            .tolist()
        )

        params = numpy.array(params).astype(float)
        if numpy.isnan(params).any():
            raise ValueError("some parameters are NaN")
        return params

    def _a_beta_from_params(self, params):
        """Vector of activities and MxE matrix of betas from params vector."""
        params_len = len(self.epitopes) * (1 + len(self.mutations))
        if params.shape != (params_len,):
            raise ValueError(f"invalid {params.shape=}")
        a = params[: len(self.epitopes)]
        beta = params[len(self.epitopes) :].reshape(
            len(self.mutations), len(self.epitopes)
        )
        assert a.shape == (len(self.epitopes),)
        assert beta.shape == (len(self.mutations), len(self.epitopes))
        assert (not numpy.isnan(a).any()) and (not numpy.isnan(beta).any())
        return (a, beta)

    def _muts_from_data_to_fit(self, data_to_fit):
        """Get wildtypes, sites, and mutations from ``data_to_fit``."""
        wts = {}
        mutations = collections.defaultdict(set)
        for variant in data_to_fit["aa_substitutions"]:
            for mutation in variant.split():
                wt, site, _ = self._mutparser.parse_mut(mutation)
                if site not in wts:
                    wts[site] = wt
                elif wts[site] != wt:
                    raise ValueError(f"inconsistent wildtype for site {site}")
                mutations[site].add(mutation)
        sites = tuple(natsort.natsorted(wts.keys(), alg=natsort.ns.SIGNED))
        wts = dict(natsort.natsorted(wts.items(), alg=natsort.ns.SIGNED))
        assert set(mutations.keys()) == set(sites) == set(wts)
        char_order = {c: i for i, c in enumerate(self.alphabet)}
        mutations = tuple(
            mut
            for site in sites
            for mut in sorted(mutations[site], key=lambda m: char_order[m[-1]])
        )
        return (wts, sites, mutations)

    def _muts_from_mut_escape_df(self, mut_escape_df):
        """Get wildtypes, sites, and mutations from ``mut_escape_df``."""
        wts = {}
        mutations = collections.defaultdict(set)
        for mutation in mut_escape_df["mutation"].unique():
            wt, site, _ = self._mutparser.parse_mut(mutation)
            if site not in wts:
                wts[site] = wt
            elif wts[site] != wt:
                raise ValueError(f"inconsistent wildtype for site {site}")
            mutations[site].add(mutation)
        sites = tuple(natsort.natsorted(wts.keys(), alg=natsort.ns.SIGNED))
        wts = dict(natsort.natsorted(wts.items(), alg=natsort.ns.SIGNED))
        assert set(mutations.keys()) == set(sites) == set(wts)
        char_order = {c: i for i, c in enumerate(self.alphabet)}
        mutations = tuple(
            mut
            for site in sites
            for mut in sorted(mutations[site], key=lambda m: char_order[m[-1]])
        )
        return (wts, sites, mutations)

    @property
    def activity_wt_df(self):
        r"""pandas.DataFrame: Activities :math:`a_{\rm{wt,e}}` for epitopes."""
        a, _ = self._a_beta_from_params(self._params)
        assert a.shape == (len(self.epitopes),)
        return pd.DataFrame(
            {
                "epitope": self.epitopes,
                "activity": a,
            }
        )

    @property
    def mut_escape_df(self):
        r"""pandas.DataFrame: Escape :math:`\beta_{m,e}` for each mutation."""
        _, beta = self._a_beta_from_params(self._params)
        assert beta.shape == (len(self.mutations), len(self.epitopes))
        df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "mutation": self.mutations,
                        "escape": b,
                        "epitope": e,
                    }
                )
                for e, b in zip(self.epitopes, beta.transpose())
            ],
            ignore_index=True,
        ).assign(
            site=lambda x: x["mutation"].map(lambda m: self._mutparser.parse_mut(m)[1]),
            mutant=lambda x: x["mutation"].map(
                lambda m: self._mutparser.parse_mut(m)[2]
            ),
            wildtype=lambda x: x["site"].map(self.wts),
        )[
            ["epitope", "site", "wildtype", "mutant", "mutation", "escape"]
        ]
        if self.mutations_times_seen is not None:
            df["times_seen"] = df["mutation"].map(self.mutations_times_seen)
            assert df.notnull().all().all()
        assert (df["wildtype"] != df["mutant"]).all()
        return df

    def mut_escape_site_summary_df(
        self,
        *,
        min_times_seen=1,
        mutation_whitelist=None,
        exclude_chars=frozenset(["*"]),
    ):
        """Site-level summaries of mutation escape.

        Parameters
        ----------
        min_times_seen : int
            Only include in summaries mutations seen in at least this many variants.
        mutation_whitelist : None or set
            Only include in summaries these mutations.
        exclude_chars : set or list
            Exclude mutations to these characters when calculating site summaries.
            Useful if you want to ignore stop codons (``*``), and perhaps in some
            cases also gaps (``-``).

        Returns
        -------
        pandas.DataFrame

        """
        escape_metrics = {
            "mean": pd.NamedAgg("escape", "mean"),
            "total positive": pd.NamedAgg("escape_gt_0", "sum"),
            "max": pd.NamedAgg("escape", "max"),
            "min": pd.NamedAgg("escape", "min"),
            "total negative": pd.NamedAgg("escape_lt_0", "sum"),
            "n mutations": pd.NamedAgg("mutation", "count"),
        }
        mut_df = self.mut_escape_df.query("mutant not in @exclude_chars")
        if self.mutations_times_seen is not None:
            mut_df = mut_df.query("times_seen >= @min_times_seen")
        if mutation_whitelist is not None:
            mut_df = mut_df.query("mutation in @mutation_whitelist")
        return (
            mut_df.assign(
                escape_gt_0=lambda x: x["escape"].clip(lower=0),
                escape_lt_0=lambda x: x["escape"].clip(upper=0),
            )
            .groupby(["epitope", "site", "wildtype"], as_index=False, sort=False)
            .aggregate(**escape_metrics)
        )

    def prob_escape(
        self,
        *,
        variants_df,
        concentrations=None,
    ):
        r"""Compute predicted probability of escape :math:`p_v\left(c\right)`.

        Computed using current mutation-escape values :math:`\beta_{m,e}` and
        epitope activities :math:`a_{\rm{wt},e}` stored in this
        :class:`Polyclonal` object.

        Arguments
        ---------
        variants_df : pandas.DataFrame
            Input data frame defining variants. Should have a column
            named 'aa_substitutions' that defines variants as space-delimited
            strings of substitutions (e.g., 'M1A K3T'). Should also have a
            column 'concentration' if ``concentrations=None``.
        concentrations : array-like or None
            Concentrations at which we compute probability of escape.

        Returns
        -------
        pandas.DataFrame
            Version of ``variants_df`` with columns named 'concentration'
            and 'predicted_prob_escape' giving predicted probability of escape
            :math:`p_v\left(c\right)` for each variant at each concentration.

        """
        prob_escape_col = "predicted_prob_escape"
        if prob_escape_col in variants_df.columns:
            raise ValueError(f"`variants_df` has column {prob_escape_col}")

        # add concentrations column to variants_df
        if concentrations is not None:
            if "concentration" in variants_df.columns:
                raise ValueError(
                    '`variants_df` has "concentration" column '
                    "and `concentrations` not `None`"
                )
            variants_df = pd.concat(
                [variants_df.assign(concentration=c) for c in concentrations],
                ignore_index=True,
            )

        (one_binarymap, binarymaps, cs, _, _, variants_df) = self._binarymaps_from_df(
            variants_df, False, False
        )

        p_v_c = self._compute_1d_pvs(self._params, one_binarymap, binarymaps, cs)
        assert p_v_c.shape == (len(variants_df),)
        variants_df[prob_escape_col] = p_v_c

        return variants_df

    def _check_close_activities(self):
        """Check that no two epitopes have near-identical activities."""
        a, _ = self._a_beta_from_params(self._params)
        a_sorted = numpy.sort(a)
        for a1, a2 in zip(a_sorted, a_sorted[1:]):
            if numpy.allclose(a1, a2):
                raise ValueError(
                    "Near-identical activities for two epitopes, "
                    "will cause problems in fitting. Reinitialize"
                    f" with more distinct activities:\n{a}"
                )

    def site_level_model(
        self,
        *,
        aggregate_mut_escapes="mean",
        collapse_identical_variants="mean",
    ):
        """Model with mutations collapsed at site level.

        Parameters
        ----------
        aggregate_mut_escapes : {'mean'}
            How to aggregate mutation-level escape values to site-level
            ones in ``mut_escape_df``.
        collapse_identical_variants : {"mean", "median", False}
            Same meaning as for :class:`Polyclonal` initialization.

        Returns
        -------
        :class:`Polyclonal`

        """
        if self.data_to_fit is None:
            site_data_to_fit = None
        else:
            site_data_to_fit = polyclonal.utils.site_level_variants(
                self.data_to_fit,
                original_alphabet=self.alphabet,
                letter_suffixed_sites=not self.sequential_integer_sites,
            )
        site_escape_df = (
            polyclonal.utils.site_level_variants(
                self.mut_escape_df.rename(columns={"mutation": "aa_substitutions"}),
                original_alphabet=self.alphabet,
                letter_suffixed_sites=not self.sequential_integer_sites,
            )
            .rename(columns={"aa_substitutions": "mutation"})
            .groupby(["epitope", "mutation"], as_index=False)
            .aggregate({"escape": aggregate_mut_escapes})
        )
        return Polyclonal(
            activity_wt_df=self.activity_wt_df,
            mut_escape_df=site_escape_df,
            data_to_fit=site_data_to_fit,
            alphabet=("w", "m"),
            sites=None if self.sequential_integer_sites else self.sites,
            epitope_colors=self.epitope_colors,
            collapse_identical_variants=collapse_identical_variants,
        )

    @staticmethod
    def _scaled_pseudo_huber(delta, r, calc_grad=False):
        r"""Compute scaled Pseudo-Huber loss (and potentially its gradient).

        :math:`h = \delta \left(\sqrt{1+\left(r/\delta\right)^2} - 1\right)`;
        this is actually :math:`1/\delta` times ``scipy.special.pseudo_huber``,
        and so has slope of one in the linear range.

        Parameters
        ----------
        delta : float
        r : numpy.ndarray
        calc_grad : bool

        Returns
        -------
        (h, dh)
            Arrays of same length as ``r``, if ``calc_grad=False`` then
            ``dh`` is None.

        >>> h, _ = Polyclonal._scaled_pseudo_huber(2, [1, 2, 4, 8], True)
        >>> h.round(2)
        array([0.24, 0.83, 2.47, 6.25])
        >>> err = scipy.optimize.check_grad(
        ...       lambda r: Polyclonal._scaled_pseudo_huber(2, r, False)[0],
        ...       lambda r: Polyclonal._scaled_pseudo_huber(2, r, True)[1],
        ...       [2])
        >>> err < 1e-7
        True

        """
        if delta <= 0:
            raise ValueError("PseudoHuber delta must be > 0")
        h = scipy.special.pseudo_huber(delta, r) / delta
        if calc_grad:
            dh = r / (h + delta)
        else:
            dh = None
        return h, dh

    def _loss_dloss(self, params, delta):
        r"""Loss on :math:`p_v\left(c\right)` and derivative wrt params."""
        pred_pvs, dpred_pvs_dparams = self._compute_1d_pvs(
            params, self._one_binarymap, self._binarymaps, self._cs, calc_grad=True
        )
        assert pred_pvs.shape == self._pvs.shape
        assert dpred_pvs_dparams.shape == (len(params), len(self._pvs))
        assert type(dpred_pvs_dparams) == scipy.sparse.csr_matrix
        residuals = pred_pvs - self._pvs
        loss, dloss_dr = self._scaled_pseudo_huber(delta, residuals, True)
        assert loss.shape == dloss_dr.shape == self._pvs.shape
        if self._weights is None:
            loss = loss.sum()
        else:
            assert loss.shape == self._weights.shape == dloss_dr.shape
            loss = (self._weights * loss).sum()
            dloss_dr = dloss_dr * self._weights
        dloss_dparams = dpred_pvs_dparams.dot(dloss_dr)
        assert dloss_dparams.shape == params.shape
        assert type(dloss_dparams) == numpy.ndarray
        return (loss, dloss_dparams)

    def _reg_escape(self, params, weight, delta):
        """Regularization on escape and its gradient."""
        if weight == 0:
            return (0, numpy.zeros(params.shape))
        elif weight < 0:
            raise ValueError(f"{weight=} for escape regularization not >= 0")
        _, beta = self._a_beta_from_params(params)
        h, dh = self._scaled_pseudo_huber(delta, beta, True)
        reg = h.sum() * weight
        assert dh.shape == beta.shape
        dreg = weight * numpy.concatenate([numpy.zeros(len(self.epitopes)), dh.ravel()])
        assert dreg.shape == params.shape
        assert numpy.isfinite(dreg).all()
        assert reg >= 0
        return reg, dreg

    def _reg_activity(self, params, weight, delta):
        """Regularization on activity and its gradient."""
        if weight == 0:
            return (0, numpy.zeros(params.shape))
        elif weight < 0:
            raise ValueError(f"{weight=} for activity regularization not >= 0")
        a, _ = self._a_beta_from_params(params)
        h, dh = self._scaled_pseudo_huber(delta, a, True)
        h = numpy.where(a > 0, h, 0.0)
        dh = numpy.where(a > 0, dh, 0.0)
        reg = h.sum() * weight
        assert dh.shape == a.shape == (len(self.epitopes),)
        dreg = weight * numpy.concatenate([dh, numpy.zeros(len(params) - len(a))])
        assert dreg.shape == params.shape
        assert numpy.isfinite(dreg).all()
        assert reg >= 0
        return reg, dreg

    def _reg_spread(self, params, weight):
        """Regularization on spread of escape at each site and its gradient."""
        if weight == 0:
            return (0, numpy.zeros(params.shape))
        elif weight < 0:
            raise ValueError(f"{weight=} for spread regularization not >= 0")
        _, beta = self._a_beta_from_params(params)
        assert beta.shape == (len(self.mutations), len(self.epitopes))
        reg = 0
        dreg = numpy.zeros(beta.shape)
        for siteindex in self._binary_sites.values():
            sitebetas = beta[siteindex]
            mi = sitebetas.shape[0]
            assert sitebetas.shape == (mi, len(self.epitopes))
            sitemeans = sitebetas.mean(axis=0)
            assert sitemeans.shape == (len(self.epitopes),)
            beta_minus_mean = sitebetas - sitemeans
            reg += weight * (beta_minus_mean**2).mean(axis=0).sum()
            dreg_site = 2 * weight / mi * beta_minus_mean
            assert dreg_site.shape == (mi, len(self.epitopes))
            dreg[siteindex] += dreg_site
        assert reg >= 0
        dreg = numpy.concatenate([numpy.zeros(len(self.epitopes)), dreg.ravel()])
        assert dreg.shape == params.shape
        assert numpy.isfinite(dreg).all()
        return reg, dreg

    def _reg_similarity(self, params, weight):
        """Regularization on similarity of escape across epitopes and its gradient."""
        if weight == 0 or len(self.epitopes) < 2:
            return (0, numpy.zeros(params.shape))
        elif weight < 0:
            raise ValueError(f"{weight=} for similarity regularization not >= 0")
        _, beta = self._a_beta_from_params(params)
        assert beta.shape == (len(self.mutations), len(self.epitopes))
        reg = 0
        dreg = numpy.zeros(beta.shape)
        site_norm = numpy.array(
            [
                (beta[siteindex] ** 2).sum(axis=0)
                for siteindex in self._binary_sites.values()
            ]
        )
        gram = site_norm.transpose() @ site_norm
        inner_prod = gram * (1 - numpy.eye(*gram.shape))
        reg += weight * (inner_prod.sum() / 2)
        norm_expanded = numpy.repeat(
            site_norm,
            [len(siteindex[0]) for siteindex in self._binary_sites.values()],
            axis=0,
        )
        norm_sum_over_epitopes = numpy.repeat(
            norm_expanded.sum(axis=1), len(self.epitopes), axis=0
        ).reshape(norm_expanded.shape[0], len(self.epitopes))
        dreg += (
            2
            * weight
            * (
                numpy.multiply(beta, norm_sum_over_epitopes)
                - numpy.multiply(beta, norm_expanded)
            )
        )
        assert reg >= 0
        dreg = numpy.concatenate([numpy.zeros(len(self.epitopes)), dreg.ravel()])
        assert dreg.shape == params.shape
        assert numpy.isfinite(dreg).all()
        return reg, dreg

    DEFAULT_SCIPY_MINIMIZE_KWARGS = frozendict.frozendict(
        {
            "method": "L-BFGS-B",
            "options": {
                "maxfun": 1e7,
                "maxiter": 1e6,
                "ftol": 1e-7,
            },
        }
    )
    """frozendict.frozendict: default ``scipy_minimize_kwargs`` to ``fit``."""

    def fit(
        self,
        *,
        loss_delta=0.1,
        reg_escape_weight=0.02,
        reg_escape_delta=0.1,
        reg_spread_weight=0.25,
        reg_similarity_weight=0,
        reg_activity_weight=1.0,
        reg_activity_delta=0.1,
        fit_site_level_first=True,
        scipy_minimize_kwargs=DEFAULT_SCIPY_MINIMIZE_KWARGS,
        log=None,
        logfreq=None,
    ):
        r"""Fit parameters (activities and mutation escapes) to the data.

        Requires :attr:`Polyclonal.data_to_fit` be set at initialization of
        this :class:`Polyclonal` object. After calling this method, the
        :math:`a_{\rm{wt},e}` and :math:`\beta_{m,e}` have been optimized, and
        can be accessed using other methods of the :class:`Polyclonal` object.

        Parameters
        ----------
        loss_delta : float
            Pseudo-Huber :math:`\delta` parameter for loss on
            :math:`p_v\left(c\right)` fitting.
        reg_escape_weight : float
            Strength of Pseudo-Huber regularization on :math:`\beta_{m,e}`.
        reg_escape_delta : float
            Pseudo-Huber :math:`\delta` for regularizing :math:`\beta_{m,e}`.
        reg_spread_weight : float
            Strength of regularization on variance of :math:`\beta_{m,e}`
            values at each site.
        reg_similarity_weight : float
            Strength of regularization on similarity of :math:`\beta_{m,e}`
            values at each site across epitopes. Has no effect when there is
            only one epitope.
        reg_activity_weight : float
            Strength of Pseudo-Huber regularization on :math:`a_{\rm{wt},e}`.
            Only positive values regularized.
        reg_activity_delta : float
            Pseudo-Huber :math:`\delta` for regularizing :math:`a_{\rm{wt},e}`.
        fit_site_level_first : bool
            First fit a site-level model, then use those activities /
            escapes to initialize fit of this model. Generally works better.
        scipy_minimize_kwargs : dict
            Keyword arguments passed to ``scipy.optimize.minimize``.
        log : None or writable file-like object
            Where to log output. If ``None``, use ``sys.stdout``.
        logfreq : None or int
            How frequently to write updates on fitting to ``log``.

        Return
        ------
        scipy.optimize.OptimizeResult

        """
        if self.data_to_fit is None:
            raise ValueError("cannot fit if `data_to_fit` not set")

        if log is None:
            log = sys.stdout
        if not (logfreq is None or (isinstance(logfreq, int) and logfreq > 0)):
            raise ValueError(f"{logfreq=} not an integer > 0")

        self._check_close_activities()

        if fit_site_level_first:
            if logfreq:
                log.write("# First fitting site-level model.\n")
            # get arg passed to fit: https://stackoverflow.com/a/65927265
            myframe = inspect.currentframe()
            keys, _, _, values = inspect.getargvalues(myframe)
            fit_kwargs = {key: values[key] for key in keys if key != "self"}
            fit_kwargs["fit_site_level_first"] = False
            site_model = self.site_level_model()
            site_model.fit(**fit_kwargs)
            self._params = self._params_from_dfs(
                activity_wt_df=site_model.activity_wt_df,
                mut_escape_df=(
                    site_model.mut_escape_df[["epitope", "site", "escape"]].merge(
                        self.mut_escape_df.drop(columns="escape"),
                        on=["epitope", "site"],
                        how="right",
                        validate="one_to_many",
                    )
                ),
            )

        class LossReg:
            # compute loss in class to remember last call
            def __init__(self_):
                self_.last_loss = None
                self_.last_params = None

            def loss_reg(self_, params, breakdown=False):
                if (self_.last_params is None) or (params != self_.last_params).any():
                    fitloss, dfitloss = self._loss_dloss(params, loss_delta)
                    regescape, dregescape = self._reg_escape(
                        params, reg_escape_weight, reg_escape_delta
                    )
                    regspread, dregspread = self._reg_spread(params, reg_spread_weight)
                    regsimilarity, dregsimilarity = self._reg_similarity(
                        params, reg_similarity_weight
                    )
                    regactivity, dregactivity = self._reg_activity(
                        params,
                        reg_activity_weight,
                        reg_activity_delta,
                    )
                    loss = fitloss + regescape + regspread + regsimilarity + regactivity
                    dloss = (
                        dfitloss
                        + dregescape
                        + dregspread
                        + dregsimilarity
                        + dregactivity
                    )
                    self_.last_params = params
                    self_.last_loss = (
                        loss,
                        dloss,
                        {
                            "fit_loss": fitloss,
                            "reg_escape": regescape,
                            "reg_spread": regspread,
                            "reg_similarity": regsimilarity,
                            "reg_activity": regactivity,
                        },
                    )
                return self_.last_loss if breakdown else self_.last_loss[:2]

        lossreg = LossReg()

        if logfreq:
            log.write(
                f"# Starting optimization of {len(self._params)} "
                f"parameters at {time.asctime()}.\n"
            )

            class Callback:
                # to log minimization
                def __init__(self_, interval, start):
                    self_.interval = interval
                    self_.i = 0
                    self_.start = start

                def callback(self_, params, header=False, force_output=False):
                    if force_output or (self_.i % self_.interval == 0):
                        loss, _, breakdown = lossreg.loss_reg(params, True)
                        if header:
                            cols = ["step", "time_sec", "loss", *breakdown.keys()]
                            log.write("".join("{:>13}".format(x) for x in cols) + "\n")
                        sec = time.time() - self_.start
                        log.write(
                            "".join(
                                "{:>13.5g}".format(x)
                                for x in [self_.i, sec, loss, *breakdown.values()]
                            )
                            + "\n"
                        )
                        log.flush()
                    self_.i += 1

            scipy_minimize_kwargs = dict(scipy_minimize_kwargs)
            callback_logger = Callback(logfreq, time.time())
            callback_logger.callback(self._params, header=True, force_output=True)
            scipy_minimize_kwargs["callback"] = callback_logger.callback

        opt_res = scipy.optimize.minimize(
            fun=lossreg.loss_reg,
            x0=self._params,
            jac=True,
            **scipy_minimize_kwargs,
        )
        self._params = opt_res.x
        if logfreq:
            callback_logger.callback(self._params, force_output=True)
            log.write(f"# Successfully finished at {time.asctime()}.\n")
        if not opt_res.success:
            log.write(f"# Optimization FAILED at {time.asctime()}.\n")
            raise PolyclonalFitError(f"Optimization failed:\n{opt_res}")
        return opt_res

    def activity_wt_barplot(self, **kwargs):
        r"""Bar plot of activity against each epitope, :math:`a_{\rm{wt},e}`.

        Parameters
        ----------
        **kwargs
            Keyword args for :func:`polyclonal.plot.activity_wt_barplot`.

        Returns
        -------
        altair.Chart
            Interactive plot.

        """
        kwargs["activity_wt_df"] = self.activity_wt_df
        if "epitope_colors" not in kwargs:
            kwargs["epitope_colors"] = self.epitope_colors
        return polyclonal.plot.activity_wt_barplot(**kwargs)

    def mut_escape_pdb_b_factor(
        self,
        *,
        input_pdbfile,
        chains,
        metric,
        outdir=None,
        outfile="{metric}-{epitope}.pdb",
        missing_metric=0,
        min_times_seen=1,
    ):
        r"""Create PDB files with B factors from a site's mutation escape.

        Parameters
        ----------
        input_pdbfile : str
            Path to input PDB file.
        chains : str or array-like
            Single chain or list of them to re-color.
        metric : str
            Which site-level summary metric to use. Can be any metric in
            :meth:`Polyclonal.mut_escape_site_summary_df`.
        outdir : str
            Output directory for created PDB files.
        outfile : str
            Output file name, with formatting used to replace metric and
            epitope in curly brackets.
        missing_metric : float or dict
            How do we handle sites in PDB that are missing in escape metric?
            If a float, reassign B factors for all missing sites to this value.
            If a dict, should be keyed by chain and assign all missing sites in
            each chain to indicated value.
        min_times_seen : int
            Value passed to :meth:`Polyclonal.mut_escape_site_summary_df`.

        Returns
        -------
        pandas.DataFrame
            Gives name of created B-factor re-colored PDB for each epitope.

        """
        df = self.mut_escape_site_summary_df(min_times_seen=min_times_seen)
        if (metric in df.columns) and (metric not in {"epitope", "site", "wildtype"}):
            metric_col = metric
        if isinstance(chains, str) and len(chains) == 1:
            chains = [chains]
        df = pd.concat([df.assign(chain=chain) for chain in chains], ignore_index=True)
        result_files = []
        for epitope in self.epitopes:
            if outdir:
                output_pdbfile = os.path.join(outdir, outfile)
            else:
                output_pdbfile = outfile
            output_pdbfile = output_pdbfile.format(
                epitope=epitope, metric=metric
            ).replace(" ", "_")
            if os.path.dirname(output_pdbfile):
                os.makedirs(os.path.dirname(output_pdbfile), exist_ok=True)
            result_files.append((epitope, output_pdbfile))
            polyclonal.pdb_utils.reassign_b_factor(
                input_pdbfile,
                output_pdbfile,
                df.query("epitope == @epitope"),
                metric_col,
                missing_metric=missing_metric,
            )
        return pd.DataFrame(result_files, columns=["epitope", "PDB file"])

    def mut_escape_plot(
        self,
        *,
        biochem_order_aas=True,
        prefix_epitope=None,
        df_to_merge=None,
        **kwargs,
    ):
        r"""Make plot of the mutation escape values, :math:`\beta_{m,e}`.

        Parameters
        ----------
        biochem_order_aas : bool
            Biochemically order the amino-acid alphabet in :attr:`Polyclonal.alphabet`
            by passing it through :func:`polyclonal.alphabets.biochem_order_aas`.
        prefix_epitope : bool or None
            Do we add the prefix "epitope " to the epitope labels? If `None`, do
            only if epitope is integer.
        df_to_merge : None or pandas.DataFrame or list
            To include additional properties, specify data frame or list of them which
            are merged with :attr:`Polyclonal.mut_escape_df` before being passed
            to :func:`polyclonal.plot.lineplot_and_heatmap`. Properties will
            only be included in plot if relevant columns are passed to
            :func:`polyclonal.plot.lineplot_and_heatmap` via `addtl_slider_stats`,
            `addtl_tooltip_stats`, or `site_zoom_bar_color_col`.
        **kwargs
            Keyword args for :func:`polyclonal.plot.lineplot_and_heatmap`.

        Returns
        -------
        altair.Chart
            Interactive line plot and heatmap.

        """
        kwargs["data_df"] = pd.concat(
            [
                self.mut_escape_df,
                (
                    self.mut_escape_df[["site", "wildtype", "epitope"]]
                    .drop_duplicates()
                    .assign(escape=0, mutant=lambda x: x["wildtype"])
                ),
            ],
        )

        if df_to_merge is not None:
            if isinstance(df_to_merge, pd.DataFrame):
                df_to_merge = [df_to_merge]
            elif not isinstance(df_to_merge, list):
                raise ValueError("`df_to_merge` must be pandas.DataFrame or list")
            for df in df_to_merge:
                if not self.sequential_integer_sites and "site" in df.columns:
                    df = df.assign(site=lambda x: x["site"].astype(str))
                kwargs["data_df"] = kwargs["data_df"].merge(df, how="left")

        if "category_colors" not in kwargs:
            kwargs["category_colors"] = self.epitope_colors

        if prefix_epitope or (
            prefix_epitope is None
            and all(type(e) == int or e.isnumeric() for e in self.epitopes)
        ):
            prefixed = {e: f"epitope {e}" for e in self.epitopes}
            kwargs["data_df"]["epitope"] = kwargs["data_df"]["epitope"].map(prefixed)
            kwargs["category_colors"] = {
                prefixed[e]: color for e, color in kwargs["category_colors"].items()
            }

        kwargs["stat_col"] = "escape"
        kwargs["category_col"] = "epitope"

        if self.mutations_times_seen:
            if "addtl_slider_stats" in kwargs:
                if "times_seen" not in kwargs["addtl_slider_stats"]:
                    kwargs["addtl_slider_stats"]["times_seen"] = 1
            else:
                kwargs["addtl_slider_stats"] = {"times_seen": 1}

        if "sites" not in kwargs:
            kwargs["sites"] = self.sites

        if "alphabet" not in kwargs:
            kwargs["alphabet"] = self.alphabet
        if biochem_order_aas:
            kwargs["alphabet"] = polyclonal.alphabets.biochem_order_aas(
                kwargs["alphabet"]
            )

        return polyclonal.plot.lineplot_and_heatmap(**kwargs)

    def filter_variants_by_seen_muts(
        self,
        variants_df,
        min_times_seen=1,
        subs_col="aa_substitutions",
    ):
        """Remove variants that contain mutations not seen during model fitting.

        Parameters
        ----------
        variants_df : pandas.DataFrame
            Contains variants as rows.
        min_times_seen : int
            Require mutations to be seen >= this many times in data used to fit model.
        subs_col : str
            Column in `variants_df` with mutations in each variant.

        Returns
        -------
        variants_df : pandas.DataFrame
            Copy of input dataframe, with rows of variants
            that have unseen mutations removed.
        """
        variants_df = variants_df.copy()

        if subs_col not in variants_df.columns:
            raise ValueError(f"`variants_df` lacks column {subs_col}")

        filter_col = "_pass_filter"
        if filter_col in variants_df.columns:
            raise ValueError(f"`variants_df` cannot have column {filter_col}")

        if min_times_seen == 1:
            allowed_muts = self.mutations
        elif self.mutations_times_seen is not None:
            allowed_muts = {
                m for (m, n) in self.mutations_times_seen.items() if n >= min_times_seen
            }
        else:
            raise ValueError(f"Cannot use {min_times_seen=} without data to fit")

        variants_df[filter_col] = variants_df[subs_col].map(
            lambda s: set(s.split()).issubset(allowed_muts)
        )

        return (
            variants_df.query("_pass_filter == True")
            .drop(columns="_pass_filter")
            .reset_index(drop=True)
        )

    def icXX(self, variants_df, *, x=0.5, col="IC50", min_c=1e-5, max_c=1e5):
        """Concentration at which a given fraction is neutralized (eg, IC50).

        Parameters
        ----------
        variants_df : pandas.DataFrame
            Data frame defining variants. Should have column named
            'aa_substitutions' that defines variants as space-delimited
            strings of substitutions (e.g., 'M1A K3T').
        x : float
            Compute concentration at which this fraction is neutralized for
            each variant. So set to 0.5 for IC50, and 0.9 for IC90.
        col : str
            Name of column in returned data frame with the ICXX value.
        min_c : float
            Minimum allowed icXX, truncate values < this at this.
        max_c : float
            Maximum allowed icXX, truncate values > this at this.

        Returns
        -------
        pandas.DataFrame
            Copy of ``variants_df`` with added column ``col`` containing icXX.

        """
        if not (0 < x < 1):
            raise ValueError(f"{x=} not >0 and <1")
        if col in variants_df.columns:
            raise ValueError(f"`variants_df` cannot have {col=}")

        reduced_df = variants_df[["aa_substitutions"]].drop_duplicates()
        bmap = self._get_binarymap(reduced_df)
        a, beta = self._a_beta_from_params(self._params)
        exp_phi_e_v = numpy.exp(-bmap.binary_variants.dot(beta) + a)
        assert exp_phi_e_v.shape == (bmap.nvariants, len(self.epitopes))
        variants = reduced_df["aa_substitutions"].tolist()
        assert len(variants) == exp_phi_e_v.shape[0]

        records = []
        for variant, exp_phi_e in zip(variants, exp_phi_e_v):
            assert exp_phi_e.shape == (len(self.epitopes),)

            def _func(c, expterm):
                pv = numpy.prod(1.0 / (1.0 + c * expterm))
                return 1 - x - pv

            if _func(min_c, exp_phi_e) > 0:
                ic = min_c
            elif _func(max_c, exp_phi_e) < 0:
                ic = max_c
            else:
                sol = scipy.optimize.root_scalar(
                    _func,
                    args=(exp_phi_e,),
                    x0=1,
                    bracket=(min_c, max_c),
                    method="brenth",
                )
                ic = sol.root
            if not sol.converged:
                raise ValueError(f"root finding failed:\n{sol}")
            records.append((variant, ic))

        ic_df = pd.DataFrame.from_records(records, columns=["aa_substitutions", col])

        return_df = variants_df.merge(
            ic_df, on="aa_substitutions", validate="many_to_one"
        )
        assert len(return_df) == len(variants_df)
        return return_df

    def _compute_1d_pvs(self, params, one_binarymap, binarymaps, cs, calc_grad=False):
        r"""Get 1D raveled array of :math:`p_v\left(c\right)` values.

        Differs from :meth:`Polyclonal._compute_pv` in that it works if just
        one or multiple BinaryMap objects.

        If `calc_grad` is `True`, also returns `scipy.sparse.csr_matrix`
        of gradient as described in :meth:`Polyclonal._compute_pv`.

        """
        if one_binarymap:
            tup = self._compute_pv(params, binarymaps, cs, calc_grad=calc_grad)
            p_vc = tup[0] if calc_grad else tup
            n_vc = binarymaps.nvariants * len(cs)
            assert p_vc.shape == (n_vc,), f"{p_vc.shape=}, {n_vc=}"
            if calc_grad:
                dpvc_dparams = tup[1]
                assert dpvc_dparams.shape == (len(params), n_vc)
                return (p_vc, dpvc_dparams)
            else:
                return p_vc
        else:
            assert len(cs) == len(binarymaps)
            p_vc = []
            dpvc_dparams = []
            n_vc = 0
            for c, bmap in zip(cs, binarymaps):
                n_vc += bmap.nvariants
                tup = self._compute_pv(
                    params, bmap, numpy.array([c]), calc_grad=calc_grad
                )
                p_vc.append(tup[0] if calc_grad else tup)
                if calc_grad:
                    dpvc_dparams.append(tup[1])
            p_vc = numpy.concatenate(p_vc)
            assert p_vc.shape == (n_vc,)
            if calc_grad:
                dpvc_dparams = scipy.sparse.hstack(dpvc_dparams).tocsr()
                assert dpvc_dparams.shape == (len(params), n_vc)
                return (p_vc, dpvc_dparams)
            else:
                return p_vc

    def _compute_pv(self, params, bmap, cs, calc_grad=False):
        r"""Compute :math:`p_v\left(c\right)` and its derivative.

        Parameters
        ----------
        params : numpy.ndarray
        bmap : binarymap.BinaryMap
        cs : numpy.ndarray
        calc_grad : bool

        Returns
        -------
        p_vc, dpvc_dparams
            ``p_vc`` is 1D array ordered by concentration and then variant
            variant. So elements are `ivariant + iconcentration * nvariants`,
            and length is nconcentrations * nvariants. If ``calc_grad=True``,
            then ``dpvc_dparams`` is `scipy.sparse.csr_matrix` of shape
            (len(params), nconcentrations * nvariants). Note that
            len(params) is nepitopes * (1 + binarylength).

        """
        a, beta = self._a_beta_from_params(params)
        assert a.shape == (len(self.epitopes),)
        assert beta.shape == (bmap.binarylength, len(self.epitopes))
        assert beta.shape[0] == bmap.binary_variants.shape[1]
        assert bmap.binary_variants.shape == (bmap.nvariants, bmap.binarylength)
        assert (cs > 0).all()
        assert cs.ndim == 1
        phi_e_v = bmap.binary_variants.dot(beta) - a
        assert phi_e_v.shape == (bmap.nvariants, len(self.epitopes))
        exp_minus_phi_e_v = numpy.exp(-phi_e_v)
        U_v_e_c = 1.0 / (1.0 + numpy.multiply.outer(exp_minus_phi_e_v, cs))
        assert U_v_e_c.shape == (bmap.nvariants, len(self.epitopes), len(cs))
        n_vc = bmap.nvariants * len(cs)
        U_vc_e = numpy.moveaxis(U_v_e_c, 1, 2).reshape(
            n_vc, len(self.epitopes), order="F"
        )
        assert U_vc_e.shape == (n_vc, len(self.epitopes))
        p_vc = U_vc_e.prod(axis=1)
        assert p_vc.shape == (n_vc,)
        if calc_grad:
            dpvc_da = p_vc * (numpy.swapaxes(U_vc_e, 0, 1) - 1)
            assert dpvc_da.shape == (len(self.epitopes), n_vc)
            dpevc = -dpvc_da.ravel(order="C")
            n_vce = n_vc * len(self.epitopes)
            assert dpevc.shape == (n_vce,)
            # Stack then transpose C X E binary_variants to multiply dpvce
            # Stacking should be fast: https://stackoverflow.com/a/45990096
            # Note after transpose this yields CSC matrix
            stacked_binary_variants = scipy.sparse.vstack(
                [bmap.binary_variants] * len(cs) * len(self.epitopes)
            ).transpose()
            assert stacked_binary_variants.shape == (bmap.binarylength, n_vce)
            dpevc_dbeta = stacked_binary_variants.multiply(
                numpy.broadcast_to(dpevc, (bmap.binarylength, n_vce))
            )
            assert dpevc_dbeta.shape == (bmap.binarylength, n_vce)
            # in params, betas sorted first by mutation, then by epitope;
            # dpevc_dbeta sorted by concentration, then variant, then epitope
            dpvc_dbetaparams = dpevc_dbeta.reshape(
                bmap.binarylength * len(self.epitopes), n_vc
            )
            assert type(dpvc_dbetaparams) == scipy.sparse.coo_matrix
            # combine to make dpvc_dparams, noting activities before betas
            # in params
            dpvc_dparams = scipy.sparse.vstack(
                [scipy.sparse.csr_matrix(dpvc_da), dpvc_dbetaparams.tocsr()]
            )
            assert type(dpvc_dparams) == scipy.sparse.csr_matrix
            assert dpvc_dparams.shape == (len(params), n_vc)
            return p_vc, dpvc_dparams
        return p_vc

    def _get_binarymap(
        self,
        variants_df,
    ):
        """Get ``BinaryMap`` appropriate for use."""
        bmap = binarymap.BinaryMap(
            variants_df,
            substitutions_col="aa_substitutions",
            allowed_subs=self.mutations,
            alphabet=self.alphabet,
            sites_as_str=not self.sequential_integer_sites,
        )
        if tuple(bmap.all_subs) != self.mutations:
            raise ValueError(
                "Different mutations in BinaryMap and self:"
                f"\n{bmap.all_subs=}\n{self.mutations=}"
            )
        return bmap

    def mut_escape_corr(self, ref_poly):
        """Correlation of mutation-escape values with another model.

        For each epitope, how well is this model's mutation-escape values
        correlation with another model?

        Mutations present in only one model are ignored.

        Parameters
        ------------
        ref_poly : :class:`Polyclonal`
            Other (reference) polyclonal model with which we calculate correlations.

        Returns
        ---------
        corr_df : pandas.DataFrame
            Pairwise epitope correlations for escape.
        """
        if self.mut_escape_df is None or ref_poly.mut_escape_df is None:
            raise ValueError("Both objects must have `mut_escape_df` initialized.")

        df = pd.concat(
            [
                self.mut_escape_df.assign(
                    epitope=lambda x: list(zip(itertools.repeat("self"), x["epitope"]))
                ),
                ref_poly.mut_escape_df.assign(
                    epitope=lambda x: list(zip(itertools.repeat("ref"), x["epitope"]))
                ),
            ]
        )

        corr = (
            polyclonal.utils.tidy_to_corr(
                df,
                sample_col="epitope",
                label_col="mutation",
                value_col="escape",
            )
            .assign(
                ref_epitope=lambda x: x["epitope_2"].map(lambda tup: tup[1]),
                ref_model=lambda x: x["epitope_2"].map(lambda tup: tup[0]),
                self_epitope=lambda x: x["epitope_1"].map(lambda tup: tup[1]),
                self_model=lambda x: x["epitope_1"].map(lambda tup: tup[0]),
            )
            .query("ref_model != self_model")[
                ["ref_epitope", "self_epitope", "correlation"]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        return corr

    def epitope_harmonized_model(self, ref_poly):
        """Get a copy of model with epitopes "harmonized" with a reference model.

        Epitopes are unidentifiable, meaning there is no guarantee that we will assign
        the same epitope labels across multiple models fit to similar data. This
        function returns a copy of the current model where the epitope labels are
        harmonized with those of another model with the same epitope labels. The
        harmonization is done by putting into correspondence the epitopes with
        the best correlated mutation-escape values.

        Parameters
        -----------
        ref_poly : :class:`Polyclonal`
            The reference polyclonal object to harmonize epitope labels with.

        Returns
        --------
        harmonized_model, harmonize_df : tuple
            `harmonized_model` is a :class:`Polyclonal` object that is a copy of
            the current model but with epitopes harmonized, and
            :attr:`Polyclonal.epitope_colors` taken from `ref_poly`. `harmonize_df`
            is a `pandas.DataFrame` that shows how the epitopes were re-assigned
            and the correlations in their escape values.

        Raises
        ------
        :class:`PolyclonalHarmonizeError`
            Raise this error if epitopes cannot be harmonized one-to-one.

        """
        if set(self.epitopes) != set(ref_poly.epitopes):
            raise PolyclonalHarmonizeError(
                "The models being harmonized have different epitope labels:\n"
                f"{self.epitopes=}\n{ref_poly.epitopes=}"
            )

        if self.sequential_integer_sites != ref_poly.sequential_integer_sites:
            raise PolyclonalHarmonizeError(
                "Models being harmonized don't have same `sequential_integer_sites`"
            )

        corr_df = (
            self.mut_escape_corr(ref_poly)
            .sort_values(["self_epitope", "correlation"], ascending=[True, False])
            .reset_index(drop=True)
        )

        harmonize_df = (
            corr_df.rename(columns={"self_epitope": "self_initial_epitope"})
            .groupby("self_initial_epitope", as_index=False)
            .first()  # will be row with highest correlation after sorting above
            .assign(self_harmonized_epitope=lambda x: x["ref_epitope"])[
                [
                    "self_initial_epitope",
                    "self_harmonized_epitope",
                    "ref_epitope",
                    "correlation",
                ]
            ]
        )
        assert len(harmonize_df) == len(self.epitopes) == len(ref_poly.epitopes)
        if not (
            set(self.epitopes)
            == set(harmonize_df["self_initial_epitope"])
            == set(harmonize_df["self_harmonized_epitope"])
        ):
            raise PolyclonalHarmonizeError(
                f"cannot harmonize 1-to-1:\n{corr_df=}\n{harmonize_df=}"
            )

        map_dict = harmonize_df.set_index("self_initial_epitope")[
            "self_harmonized_epitope"
        ].to_dict()
        assert len(map_dict) == len(self.epitopes) == len(set(map_dict.values()))

        # for sorting epitopes
        epitope_order = {e: i for i, e in enumerate(ref_poly.epitopes)}

        harmonized_model = Polyclonal(
            activity_wt_df=(
                self.activity_wt_df.assign(epitope=lambda x: x["epitope"].map(map_dict))
                .sort_values("epitope", key=lambda c: c.map(epitope_order))
                .reset_index(drop=True)
            ),
            mut_escape_df=(
                self.mut_escape_df.assign(epitope=lambda x: x["epitope"].map(map_dict))
                .sort_values("epitope", key=lambda c: c.map(epitope_order))
                .reset_index(drop=True)
            ),
            data_to_fit=self.data_to_fit,
            collapse_identical_variants=False,  # already collapsed if this done for self
            alphabet=self.alphabet,
            epitope_colors=ref_poly.epitope_colors,
            data_mut_escape_overlap="exact_match",  # should be exact match in self
            sites=None if self.sequential_integer_sites else self.sites,
        )
        assert ref_poly.epitopes == harmonized_model.epitopes
        return harmonized_model, harmonize_df


if __name__ == "__main__":
    import doctest

    doctest.testmod()
