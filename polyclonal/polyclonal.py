"""
==========
polyclonal
==========

Defines :class:`Polyclonal` objects for handling antibody mixtures.

"""


import collections
import inspect
import itertools
import os
import sys
import time


import binarymap

import frozendict

import numpy

import pandas as pd

import scipy.optimize
import scipy.special

import polyclonal.pdb_utils
import polyclonal.plot
import polyclonal.utils


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
        variants. Collapse by taking mean or median of 'prob_escape', or
        do not collapse at all.
    alphabet : array-like
        Allowed characters in mutation strings.
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
    data_mut_escape_overlap : {'exact_match', 'fill_to_data'}
        If ``data_to_fit`` and ``mut_escape_df`` (or ``site_escape_df``) both
        specificied, what if they don't specify same mutations.
        If 'exact_match', raise error. If 'fill_to_data', then take
        sites / wildtypes / mutations from ``data_to_fit`` and fill init
        values from any not specified in ``mut_escape_df`` as indicated by
        ``init_missing``--still raise error if values in ``mut_escape_df``
        are not in ``data_to_fit``.

    Attributes
    ----------
    epitopes : tuple
        Names of all epitopes.
    mutations : tuple
        All mutations for which we have escape values.
    alphabet : tuple
        Allowed characters in mutation strings.
    sites : tuple
        List of all sites.
    wts : dict
        Keyed by site, value is wildtype at that site.
    epitope_colors : dict
        Maps each epitope to its color.
    data_to_fit : pandas.DataFrame or None
        Data to fit as passed when initializing this :class:`BinaryMap`.
        If using ``collapse_identical_variants``, then identical variants
        are collapsed on columns 'concentration', 'aa_substitutions',
        and 'prob_escape', and a column 'weight' is added to represent number
        of collapsed variants. Also, row-order may be changed.

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
    >>> polyclonal = Polyclonal(activity_wt_df=activity_wt_df,
    ...                         mut_escape_df=mut_escape_df)
    >>> polyclonal.epitopes
    ('e1', 'e2')
    >>> polyclonal.mutations
    ('M1C', 'G2A', 'A4K', 'A4L')
    >>> polyclonal.sites
    (1, 2, 4)
    >>> polyclonal.wts
    {1: 'M', 2: 'G', 4: 'A'}
    >>> polyclonal.activity_wt_df
      epitope  activity
    0      e1       2.0
    1      e2       1.0
    >>> polyclonal.mut_escape_df
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

    >>> polyclonal.mut_escape_site_summary_df
      epitope  site wildtype  mean  total positive  max  min  total negative
    0      e1     1        M   2.0             2.0  2.0  2.0             0.0
    1      e1     2        G   3.0             3.0  3.0  3.0             0.0
    2      e1     4        A   0.0             0.0  0.0  0.0             0.0
    3      e2     1        M   0.0             0.0  0.0  0.0             0.0
    4      e2     2        G   0.0             0.0  0.0  0.0             0.0
    5      e2     4        A   2.0             4.0  2.5  1.5             0.0

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

    >>> escape_probs = polyclonal.prob_escape(variants_df=variants_df,
    ...                                       concentrations=[1.0, 2.0, 4.0])
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

    >>> polyclonal.prob_escape(
    ...         variants_df=pd.concat([variants_df.assign(concentration=c)
    ...                                for c in [1.0, 2.0, 4.0]])
    ...         ).equals(escape_probs)
    True

    We can also compute the IC50s:

    >>> polyclonal.icXX(variants_df).round(3)
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

    >>> polyclonal.icXX(variants_df, x=0.9, col='IC90').round(3)
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

    >>> polyclonal_data = Polyclonal(data_to_fit=data_to_fit,
    ...                              n_epitopes=2)

    The mutations are those in ``data_to_fit``:

    >>> polyclonal_data.mutations
    ('M1C', 'G2A', 'A4K', 'A4L')

    The activities are evenly spaced from 1 to 0, while the mutation escapes
    are all initialized to zero:

    >>> polyclonal_data.activity_wt_df
      epitope  activity
    0       1       1.0
    1       2       0.0
    >>> polyclonal_data.mut_escape_df
      epitope  site wildtype mutant mutation  escape
    0       1     1        M      C      M1C     0.0
    1       1     2        G      A      G2A     0.0
    2       1     4        A      K      A4K     0.0
    3       1     4        A      L      A4L     0.0
    4       2     1        M      C      M1C     0.0
    5       2     2        G      A      G2A     0.0
    6       2     4        A      K      A4K     0.0
    7       2     4        A      L      A4L     0.0

    You can initialize to random numbers by setting ``init_missing`` to seed
    (in this example we also don't include all variants for one concentration):

    >>> polyclonal_data2 = Polyclonal(data_to_fit=data_to_fit.head(30),
    ...                               n_epitopes=2,
    ...                               init_missing=1,
    ...                               )
    >>> polyclonal_data2.activity_wt_df.round(3)
      epitope  activity
    0       1     0.417
    1       2     0.720

    You can set some or all mutation escapes to initial values:

    >>> polyclonal_data3 = Polyclonal(
    ...            data_to_fit=data_to_fit,
    ...            activity_wt_df=activity_wt_df,
    ...            mut_escape_df=pd.DataFrame({'epitope': ['e1'],
    ...                                        'mutation': ['M1C'],
    ...                                        'escape': [4]}),
    ...            data_mut_escape_overlap='fill_to_data',
    ...            )
    >>> polyclonal_data3.mut_escape_df
      epitope  site wildtype mutant mutation  escape
    0      e1     1        M      C      M1C     4.0
    1      e1     2        G      A      G2A     0.0
    2      e1     4        A      K      A4K     0.0
    3      e1     4        A      L      A4L     0.0
    4      e2     1        M      C      M1C     0.0
    5      e2     2        G      A      G2A     0.0
    6      e2     4        A      K      A4K     0.0
    7      e2     4        A      L      A4L     0.0

    You can initialize **sites** to escape values via ``site_activity_df``:

    >>> polyclonal_data4 = Polyclonal(
    ...        data_to_fit=data_to_fit,
    ...        activity_wt_df=activity_wt_df,
    ...        site_escape_df=pd.DataFrame.from_records(
    ...                [('e1', 1, 1.0), ('e1', 4, 0.0),
    ...                 ('e2', 1, 0.0), ('e2', 4, 2.0)],
    ...                columns=['epitope', 'site', 'escape']),
    ...        data_mut_escape_overlap='fill_to_data',
    ...        )
    >>> polyclonal_data4.mut_escape_df
      epitope  site wildtype mutant mutation  escape
    0      e1     1        M      C      M1C     1.0
    1      e1     2        G      A      G2A     0.0
    2      e1     4        A      K      A4K     0.0
    3      e1     4        A      L      A4L     0.0
    4      e2     1        M      C      M1C     0.0
    5      e2     2        G      A      G2A     0.0
    6      e2     4        A      K      A4K     2.0
    7      e2     4        A      L      A4L     2.0

    Fit the data using :meth:`Polyclonal.fit`, and make sure the new
    predicted escape probabilities are close to the real ones being fit.
    Reduce weight on regularization since there is so little data in this
    toy example:

    >>> for model in [polyclonal_data, polyclonal_data2,
    ...               polyclonal_data3, polyclonal_data4]:
    ...     opt_res = model.fit(reg_escape_weight=0.001,
    ...                         reg_spread_weight=0.001)
    ...     pred_df = model.prob_escape(variants_df=data_to_fit)
    ...     if not numpy.allclose(pred_df['prob_escape'],
    ...                           pred_df['predicted_prob_escape'],
    ...                           atol=0.01):
    ...          raise ValueError(f"wrong predictions\n{pred_df}")
    ...     if not numpy.allclose(
    ...              activity_wt_df['activity'].sort_values(),
    ...              model.activity_wt_df['activity'].sort_values(),
    ...              atol=0.1,
    ...              ):
    ...          raise ValueError(f"wrong activities\n{model.activity_wt_df}")
    ...     if not numpy.allclose(
    ...              mut_escape_df['escape'].sort_values(),
    ...              model.mut_escape_df['escape'].sort_values(),
    ...              atol=0.05,
    ...              ):
    ...          raise ValueError(f"wrong escapes\n{model.mut_escape_df}")

    Example
    -------
    You can convert a :class:`Polyclonal` model into a site-level model via
    the transformation of :func:`polyclonal.utils.site_level_variants`. The
    site-level model is another :class:`Polyclonal` model that just keeps
    track of whether or not sites are mutated using a 2-letter wildtype/mutant
    alphabet, and is generated using :meth:`Polyclonal.site_level_model`:

    >>> polyclonal_site = polyclonal_data4.site_level_model()
    >>> polyclonal_site.alphabet
    ('w', 'm')
    >>> (polyclonal_site.mut_escape_df
    ...  .assign(escape=lambda x: x['escape'].abs()).round(1))
      epitope  site wildtype mutant mutation  escape
    0      e1     1        w      m      w1m     2.0
    1      e1     2        w      m      w2m     3.0
    2      e1     4        w      m      w4m     0.0
    3      e2     1        w      m      w1m     0.0
    4      e2     2        w      m      w2m     0.0
    5      e2     4        w      m      w4m     2.0
    >>> polyclonal_site.data_to_fit.head(n=5).round(3)
       concentration aa_substitutions  weight  prob_escape
    0            1.0                        1        0.032
    1            1.0              w1m       1        0.134
    2            1.0          w1m w2m       1        0.256
    3            1.0      w1m w2m w4m       2        0.686
    4            1.0          w1m w4m       1        0.409

    """

    def __init__(
        self,
        *,
        activity_wt_df=None,
        mut_escape_df=None,
        data_to_fit=None,
        site_escape_df=None,
        n_epitopes=None,
        collapse_identical_variants="mean",
        alphabet=binarymap.binarymap.AAS_NOSTOP,
        epitope_colors=polyclonal.plot.TAB10_COLORS_NOGRAY,
        init_missing="zero",
        data_mut_escape_overlap="exact_match",
    ):
        """See main class docstring."""
        if isinstance(init_missing, int):
            numpy.random.seed(init_missing)
        elif init_missing != "zero":
            raise ValueError(f"invalid {init_missing=}")

        if len(set(alphabet)) != len(alphabet):
            raise ValueError("duplicate letters in `alphabet`")
        self.alphabet = tuple(alphabet)
        self._mutparser = polyclonal.utils.MutationParser(alphabet)

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
            if not isinstance(n_epitopes, int) and n_epitopes > 0:
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
            self.epitope_colors = {epitope_colors[e] for e in self.epitopes}
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
        if site_escape_df is not None:
            # construct mut_escape_df from site_escape_df and mutations
            # from data_to_fit
            req_cols = {"epitope", "site", "escape"}
            if not req_cols.issubset(site_escape_df.columns):
                raise ValueError(f"`site_escape_df` lacks columns {req_cols}")
            assert (data_to_fit is not None) and (mut_escape_df is None)
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
        if mut_escape_df is data_to_fit is None:
            raise ValueError("initialize `mut_escape_df` or `data_to_fit`")
        elif mut_escape_df is None:
            self.wts, self.sites, self.mutations = wts2, sites2, muts2
            mut_escape_df = _init_mut_escape_df(self.mutations)
        elif data_to_fit is None:
            self.wts, self.sites, self.mutations = wts, sites, muts
        else:
            if data_mut_escape_overlap == "exact_match":
                if sites == sites2 and wts == wts2 and muts == muts2:
                    self.wts, self.sites, self.mutations = wts, sites, muts
                else:
                    raise ValueError(
                        "`data_to_fit` and `mut_escape_df` give "
                        "different mutations. Fix or set "
                        'data_mut_escape_overlap="fill_to_data"'
                    )
            elif data_mut_escape_overlap == "fill_to_data":
                if set(sites) < set(sites2):
                    self.sites = sites2
                else:
                    raise ValueError(
                        "`mut_escape_df` has more sites than " "`data_to_fit`"
                    )
                if wts.items() <= wts2.items():
                    self.wts = wts2
                else:
                    raise ValueError("`mut_escape_df` has wts not in " "`data_to_fit`")
                if set(muts) < set(muts2):
                    self.mutations = muts2
                else:
                    raise ValueError(
                        "`mut_escape_df` has mutations not in " "`data_to_fit`"
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
        if get_pv:
            cols.append("prob_escape")
        if not df[cols].notnull().all().all():
            raise ValueError(f"null entries in data frame of variants:\n{df}")
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
        sites = tuple(sorted(wts.keys()))
        wts = dict(sorted(wts.items()))
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
        sites = tuple(sorted(wts.keys()))
        wts = dict(sorted(wts.items()))
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
        return pd.concat(
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

    @property
    def mut_escape_site_summary_df(self):
        r"""pandas.DataFrame: Site-level summaries of mutation escape."""
        escape_metrics = {
            "mean": pd.NamedAgg("escape", "mean"),
            "total positive": pd.NamedAgg("escape_gt_0", "sum"),
            "max": pd.NamedAgg("escape", "max"),
            "min": pd.NamedAgg("escape", "min"),
            "total negative": pd.NamedAgg("escape_lt_0", "sum"),
        }
        return (
            self.mut_escape_df.assign(
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

        Computed using current mutation-escape values :math:`beta_{m,e}` and
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
    ):
        """Model with mutations collapsed at site level.

        Parameters
        ----------
        aggregate_mut_escapes : {'mean'}
            How to aggregate mutation-level escape values to site-level
            ones in ``mut_effects_df``.

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
            )
        site_escape_df = (
            polyclonal.utils.site_level_variants(
                self.mut_escape_df.rename(columns={"mutation": "aa_substitutions"})
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
            epitope_colors=self.epitope_colors,
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
        reg_escape_weight=0.01,
        reg_escape_delta=0.1,
        reg_spread_weight=0.25,
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
        fit_site_level_first : bool
            First fit a site-level model, then use those activities /
            escapes to initialize fit of this model. Generally works better.
        scipy_minimize_kwargs : dict
            Keyword arguments passed to ``scipy.optimize.minimize``.
        log : None or writable file-like object
            Where to log output if ``verbosity > 0``. If ``None``, use
            ``sys.stdout``.
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
                    loss = fitloss + regescape + regspread
                    dloss = dfitloss + dregescape + dregspread
                    self_.last_params = params
                    self_.last_loss = (
                        loss,
                        dloss,
                        {
                            "fit_loss": fitloss,
                            "reg_escape": regescape,
                            "regspread": regspread,
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
                            log.write("".join("{:>11}".format(x) for x in cols) + "\n")
                        sec = time.time() - self_.start
                        log.write(
                            "".join(
                                "{:>11.5g}".format(x)
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
            raise RuntimeError(f"Optimization failed:\n{opt_res}")
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
            :attr:`Polyclonal.mut_escape_site_summary_df`.
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

        Returns
        -------
        pandas.DataFrame
            Gives name of created B-factor re-colored PDB for each epitope.

        """
        df = self.mut_escape_site_summary_df
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

    def mut_escape_lineplot(self, **kwargs):
        r"""Line plots of mutation escape :math:`\beta_{m,e}` at each site.

        Parameters
        ----------
        **kwargs
            Keyword args for :func:`polyclonal.plot.mut_escape_lineplot`.

        Returns
        -------
        altair.Chart
            Interactive plot.

        """
        kwargs["mut_escape_site_summary_df"] = self.mut_escape_site_summary_df
        if "epitope_colors" not in kwargs:
            kwargs["epitope_colors"] = self.epitope_colors
        return polyclonal.plot.mut_escape_lineplot(**kwargs)

    def mut_escape_heatmap(self, **kwargs):
        r"""Heatmaps of the mutation escape values, :math:`\beta_{m,e}`.

        Parameters
        ----------
        **kwargs
            Keyword args for :func:`polyclonal.plot.mut_escape_heatmap`.

        Returns
        -------
        altair.Chart
            Interactive heat maps.

        """
        kwargs["mut_escape_df"] = self.mut_escape_df
        if "epitope_colors" not in kwargs:
            kwargs["epitope_colors"] = self.epitope_colors
        if "alphabet" not in kwargs:
            kwargs["alphabet"] = self.alphabet
        return polyclonal.plot.mut_escape_heatmap(**kwargs)

    def filter_variants_by_seen_muts(self, variants_df, subs_col="aa_substitutions"):
        """Remove variants that contain mutations not seen during model fitting.

        Parameters
        ----------
        variants_df : pandas.DataFrame
            Contains variants as rows.

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

        variants_df[filter_col] = variants_df[subs_col].map(
            lambda s: set(s.split()).issubset(self.mutations)
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

            def _func(c):
                pv = numpy.prod(1.0 / (1.0 + c * exp_phi_e))
                return 1 - x - pv

            if _func(min_c) > 0:
                ic = min_c
            elif _func(max_c) < 0:
                ic = max_c
            else:
                sol = scipy.optimize.root_scalar(
                    _func, x0=1, bracket=(min_c, max_c), method="brenth"
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
        )
        assert tuple(bmap.all_subs) == self.mutations
        return bmap

    def _make_correlation_matrix(self, ref_poly):
        """Calculate the correlation of mutation-escape values of all pairwise
        combinations of `self`s model and the model inferred by `ref_poly`.
        Both inputs must have the same number of epitopes specified.

        The rows of this matrix index the epitopes in `self`, the columns do the
        same for `ref_poly`.

        In the situation where a mutation is only seen by one model, we exclude
        that mutation from correlation calculations (i.e., we omit the rows
        where either `escape_x` and `escape_y` of the merged df contain None.)

        Parameters
        ------------
        ref_poly : :class:Polyclonal object
            The reference polyclonal object to calculate the correlation of
            mutation-escape values between all pairwise combinations of
            epitopes.

        Returns
        ---------
        corr_df : pandas.DataFrame
            A dataframe object with pairwise epitope correlations for escape.
        """
        corr_df = pd.DataFrame()
        for model_ref_e, model_self_e in list(
            itertools.product(
                ref_poly.mut_escape_df["epitope"].unique(),
                self.mut_escape_df["epitope"].unique(),
            )
        ):
            df = ref_poly.mut_escape_df.query(f'epitope == "{model_ref_e}"').merge(
                (self.mut_escape_df.query(f'`epitope` == "{model_self_e}"')),
                on="mutation",
                validate="one_to_one",
            )

            corr = df["escape_x"].corr(df["escape_y"])

            corr_df = pd.concat(
                [
                    corr_df,
                    pd.DataFrame(
                        data={
                            "ref_epitope": [model_ref_e],
                            "self_epitope": [model_self_e],
                            "correlation": [corr],
                        }
                    ),
                ]
            ).reset_index(drop=True)

        return corr_df

    def _create_max_correlation_mapping_matrix(self, corr_df):
        """Create a binary matrix for mapping epitopes to one another.

        For each epitope, we align it with the reference model epitope with the
        highest correlation in escape values.

        Parameters
        -----------
        corr_df : pandas.DataFrame
            An correlation matrix between the inferred escape probabilities
            amongst the epitopes of two models.

        Returns
        --------
        map_mat : numpy.array
            A matrix of 1s and 0s depicting how to shift the epitopes of the
            non-reference axis.
        """
        corr_mat = corr_df.pivot(
            index="self_epitope", columns="ref_epitope", values="correlation"
        ).values
        map_mat = numpy.zeros_like(corr_mat)

        for idx, row in pd.DataFrame(corr_mat).iterrows():
            match_epi = row.argmax()
            map_mat[idx, match_epi] = 1

        # Check to see if epitope mapping is 1-to-1 before returning
        self._check_epitope_mapping_matrix(map_mat)

        return map_mat

    def _check_epitope_mapping_matrix(self, map_mat):
        """Check to ensure an epitope mapping matrix is 1-to-1.

        Raises a `ValueError` if all of the columns in this matrix have a column
        sum that is not one.

        Parameters
        -----------
        map_mat : numpy.array
            A square binary matrix of indicators between `self` and a
            reference polyclonal object's epitopes.
        """
        if numpy.any(map_mat.sum(axis=0) != 1):
            raise ValueError("Mapping matrix does not have a 1-to-1 mapping.")
        if map_mat.shape[0] != map_mat.shape[1]:
            raise ValueError("Mapping matrix is not square.")
        if not numpy.isin(map_mat, [0, 1]).all():
            raise ValueError("Mapping matrix has values other than 0 or 1.")

    def _make_mapping_dict(self, map_mat, ref_poly):
        """Create a dictionary from a mapping matrix to replace values in the
        `mut_escape_df` dataframe.

        Parameters
        -----------
        map_mat : numpy.array
            A 2D array from `self._create_max_correlation_mapping_matrix` that aligns
            epitopes.
        ref_poly : :class:Polyclonal
            The reference polyclonal object to align to.

        Returns
        --------
        epi_dict : Dictionary
            A dictionary of 'self:ref` key-value pairs of "harmoninzed" epitopes.
        """
        epi_dict = {}
        for i in range(len(map_mat)):
            ref_epi_idx = map_mat[i].argmax()
            epi_dict[self.epitopes[i]] = ref_poly.epitopes[ref_epi_idx]
        return epi_dict

    def _edit_epitopes_in_mut_escape_df(self, mapping_dict):
        """Create a new version of `mut_escape_df` with harmonized epitopes.

        Parameters
        -----------
        mapping_dict : Dictionary
            Dictionary of self-reference epitope IDs.

        Returns
        -------
        An edited pandas.DataFrame with edited epitopes corresponding to `mapping_dict`.
        """
        if mapping_dict is None:
            raise AttributeError("`_mapping_dict` not defined.")
        return (
            self.mut_escape_df.replace({"epitope": mapping_dict})
            .sort_values(by=["epitope", "site"])
            .reset_index(drop=True)
        )

    def _align_params(self, mapping_dict):
        """Reorders corresponding epitope parameters in `self._params`.

        Parameters
        -----------
        mapping_dict : Dictionary
            A dictionary of `self:ref` key-value pairs of "harmonized" epitopes.
        """
        # Get new ordering of epitope indicies for _params.
        new_idxs = numpy.fromiter(mapping_dict.values(), dtype=int) - 1
        if len(new_idxs) != len(self.epitopes):
            raise ValueError(
                "Number of WT activity params not equal to number of epitopes."
            )
        # This will make sure `activity_wt_df()` is aligned in future calls.
        aligned_activity_wt_vals = self._params[new_idxs]
        self._params[0 : len(self.epitopes)] = aligned_activity_wt_vals
        # Now edit mut escape df
        aligned_mut_escape_df = self._edit_epitopes_in_mut_escape_df(mapping_dict)
        # Align the params -- now mut_escape_df should be rearranged.
        self._params = self._params_from_dfs(self.activity_wt_df, aligned_mut_escape_df)

    def harmonize_epitopes_with(self, ref_poly):
        """Harmonize epitopes with another polyclonal object.
        Epitopes are unidentifiable, meaning there is no gurantee that we will infer
        the same epitopes across multiple models.
        This function aims to "align" inferred epitopes with two models.

        Parameters
        -----------
        ref_poly : :class:Polyclonal
            The reference polyclonal object to align epitopes to.

        Returns
        --------
        self : :class:Polyclonal
            Enables `_mapping_dict` so aligned properties can be returned.
        """
        # Checks to ensure `ref_poly` and self are compatible.

        # Make sure models have same number of params.
        if len(self._params) != len(ref_poly._params):
            raise ValueError(
                "The polyclonal objects have a different number of parameters."
            )

        # We must have the same number of epitopes
        if len(self.epitopes) != len(ref_poly.epitopes):
            raise ValueError(
                "The two models being aligned do not have the same number of "
                "epitopes."
            )

        # Both models should have `mut_escape_df` initialized
        if self.mut_escape_df is None or ref_poly.mut_escape_df is None:
            raise ValueError("Both objects must have `mut_escape_df` initialized.")

        # Add another check to ensure both `mut_escape_df`s have needed columns
        required_cols = {"epitope", "mutation", "escape"}
        if not required_cols.issubset(self.mut_escape_df.columns):
            raise KeyError(
                "The `mut_escape_df` of the object being aligned does not "
                f"contain all of the required columns: {required_cols}."
            )

        if not required_cols.issubset(ref_poly.mut_escape_df.columns):
            raise KeyError(
                "The `mut_escape_df` of the reference object being aligned to "
                f"does not contain all of the required columns: {required_cols}."
            )

        # Step one: get correlation matrix
        corr_df = self._make_correlation_matrix(ref_poly)

        # Step two: create mapping matrix
        mapping_mat = self._create_max_correlation_mapping_matrix(corr_df)

        # Step three: make epitope mapping dictionary
        epi_dict = self._make_mapping_dict(mapping_mat, ref_poly)

        # Align params.
        self._align_params(epi_dict)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
