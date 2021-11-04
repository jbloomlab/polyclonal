"""
==========
polyclonal
==========

Defines :class:`Polyclonal` objects for handling antibody mixtures.

"""


import collections
import os
import re

import binarymap

import numpy

import pandas as pd

import scipy.optimize

import polyclonal.pdb_utils
import polyclonal.plot


AAS_NOSTOP = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
              'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')
"""tuple: Amino-acid one-letter codes alphabetized, doesn't include stop."""


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
       and ``activity_wt_df`` and ``mut_escape_df`` holding initial guesses
       for these values, ensuring guesses in ``mut_escape_df`` encompass
       same mutations as ``data_to_fit``. Then call :meth:`Polyclonal.fit`.

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
    n_epitopes : int or None
        If initializing with ``activity_wt_df=None``, specifies number
        of epitopes.
    alphabet : array-like
        Allowed characters in mutation strings.
    epitope_colors : array-like or dict
        Maps each epitope to the color used for plotting. Either a dict keyed
        by each epitope, or an array of colors that are sequentially assigned
        to the epitopes.
    init_missing : 'zero' or int
        How to to initialize any activities or mutation-escape values not
        specified in ``activity_wt_df`` or ``mut_escape_df``. If 'zero',
        set to zero. Otherwise draw uniformly from between 0 and 1 using
        specified random number seed.
    data_mut_escape_overlap : {'exact_match', 'fill_to_data'}
        If ``data_to_fit`` and ``mut_escape_df`` are both specified,
        what to do if they don't specify same sites / wildtypes / mutations.
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
        Data to fit as passed when initializing this :class:`BinaryMap`,
        although possibly in different row order.

    Example
    -------
    Simple example with two epitopes (`e1` and `e2`) and a few mutations where
    we know the activities and mutation-level escape values ahead of time:

    >>> activity_wt_df = pd.DataFrame({'epitope':  ['e1', 'e2'],
    ...                                'activity': [ 2.0,  1.0]})
    >>> mut_escape_df = pd.DataFrame({
    ...      'mutation': ['M1C', 'M1A', 'M1A', 'M1C', 'A2K', 'A2K'],
    ...      'epitope':  [ 'e1',  'e2',  'e1',  'e2',  'e1',  'e2'],
    ...      'escape':   [  2.0,   0.0,   3.0,  0.0,   0.0,   2.5]})
    >>> polyclonal = Polyclonal(activity_wt_df=activity_wt_df,
    ...                         mut_escape_df=mut_escape_df)
    >>> polyclonal.epitopes
    ('e1', 'e2')
    >>> polyclonal.mutations
    ('M1A', 'M1C', 'A2K')
    >>> polyclonal.sites
    (1, 2)
    >>> polyclonal.wts
    {1: 'M', 2: 'A'}
    >>> polyclonal.activity_wt_df
      epitope  activity
    0      e1       2.0
    1      e2       1.0
    >>> polyclonal.mut_escape_df
      epitope  site wildtype mutant mutation  escape
    0      e1     1        M      A      M1A     3.0
    1      e1     1        M      C      M1C     2.0
    2      e1     2        A      K      A2K     0.0
    3      e2     1        M      A      M1A     0.0
    4      e2     1        M      C      M1C     0.0
    5      e2     2        A      K      A2K     2.5

    We can also summarize the mutation-level escape at the site level:

    >>> polyclonal.mut_escape_site_summary_df
      epitope  site wildtype  mean  total positive  max  min  total negative
    0      e1     1        M   2.5             5.0  3.0  2.0             0.0
    1      e1     2        A   0.0             0.0  0.0  0.0             0.0
    2      e2     1        M   0.0             0.0  0.0  0.0             0.0
    3      e2     2        A   2.5             2.5  2.5  2.5             0.0

    Note that we can **not** initialize a :class:`Polyclonal` object if we are
    missing escape estimates for any mutations for any epitopes:

    >>> Polyclonal(activity_wt_df=activity_wt_df,
    ...            mut_escape_df=mut_escape_df.head(n=5))
    Traceback (most recent call last):
      ...
    ValueError: invalid set of mutations for epitope='e2'

    Now make a data frame with some variants:

    >>> variants_df = pd.DataFrame.from_records(
    ...         [('AA', 'A2K'),
    ...          ('AC', 'M1A A2K'),
    ...          ('AG', 'M1A'),
    ...          ('AT', ''),
    ...          ('CA', 'A2K')],
    ...         columns=['barcode', 'aa_substitutions'])

    Get the escape probabilities predicted on these variants from
    the values in the :class:`Polyclonal` object:

    >>> escape_probs = polyclonal.prob_escape(variants_df=variants_df,
    ...                                       concentrations=[1.0, 2.0, 4.0])
    >>> escape_probs.round(3)
       barcode aa_substitutions  concentration  predicted_prob_escape
    0       AT                             1.0                  0.032
    1       AA              A2K            1.0                  0.097
    2       CA              A2K            1.0                  0.097
    3       AG              M1A            1.0                  0.197
    4       AC          M1A A2K            1.0                  0.598
    5       AT                             2.0                  0.010
    6       AA              A2K            2.0                  0.044
    7       CA              A2K            2.0                  0.044
    8       AG              M1A            2.0                  0.090
    9       AC          M1A A2K            2.0                  0.398
    10      AT                             4.0                  0.003
    11      AA              A2K            4.0                  0.017
    12      CA              A2K            4.0                  0.017
    13      AG              M1A            4.0                  0.034
    14      AC          M1A A2K            4.0                  0.214

    We can also get predicted escape probabilities by including concentrations
    in the data frame passed to :meth:`Polyclonal.prob_escape`:

    >>> df_with_conc = pd.concat([variants_df.assign(concentration=c)
    ...                           for c in [1.0, 2.0, 4.0]]).head(14)
    >>> polyclonal.prob_escape(variants_df=df_with_conc).round(3)
       barcode aa_substitutions  concentration  predicted_prob_escape
    0       AT                             1.0                  0.032
    1       AA              A2K            1.0                  0.097
    2       CA              A2K            1.0                  0.097
    3       AG              M1A            1.0                  0.197
    4       AC          M1A A2K            1.0                  0.598
    5       AT                             2.0                  0.010
    6       AA              A2K            2.0                  0.044
    7       CA              A2K            2.0                  0.044
    8       AG              M1A            2.0                  0.090
    9       AC          M1A A2K            2.0                  0.398
    10      AT                             4.0                  0.003
    11      AA              A2K            4.0                  0.017
    12      AG              M1A            4.0                  0.034
    13      AC          M1A A2K            4.0                  0.214

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
    ('M1A', 'A2K')

    The activities and mutation escapes are all initialized to zero:

    >>> polyclonal_data.activity_wt_df
         epitope  activity
    0  epitope 1       0.0
    1  epitope 2       0.0
    >>> polyclonal_data.mut_escape_df
         epitope  site wildtype mutant mutation  escape
    0  epitope 1     1        M      A      M1A     0.0
    1  epitope 1     2        A      K      A2K     0.0
    2  epitope 2     1        M      A      M1A     0.0
    3  epitope 2     2        A      K      A2K     0.0

    You can initialize to random numbers by setting ``init_missing`` to seed:

    >>> Polyclonal(data_to_fit=data_to_fit,
    ...            n_epitopes=2,
    ...            init_missing=1,
    ...            ).activity_wt_df.round(3)
         epitope  activity
    0  epitope 1     0.417
    1  epitope 2     0.720

    You set some or all mutation escapes to initial values:

    >>> polyclonal_data2 = Polyclonal(
    ...            data_to_fit=data_to_fit,
    ...            activity_wt_df=activity_wt_df,
    ...            mut_escape_df=pd.DataFrame({'epitope': ['e1'],
    ...                                        'mutation': ['M1A'],
    ...                                        'escape': [4]}),
    ...            data_mut_escape_overlap='fill_to_data',
    ...            )
    >>> polyclonal_data2.mut_escape_df
      epitope  site wildtype mutant mutation  escape
    0      e1     1        M      A      M1A     4.0
    1      e1     2        A      K      A2K     0.0
    2      e2     1        M      A      M1A     0.0
    3      e2     2        A      K      A2K     0.0

    Fit the values:

    >>> opt_res = polyclonal_data.fit()
    >>> opt_res

    """

    def __init__(self,
                 *,
                 activity_wt_df=None,
                 mut_escape_df=None,
                 data_to_fit=None,
                 n_epitopes=None,
                 alphabet=AAS_NOSTOP,
                 epitope_colors=polyclonal.plot.TAB10_COLORS_NOGRAY,
                 init_missing='zero',
                 data_mut_escape_overlap='exact_match',
                 ):
        """See main class docstring."""
        if isinstance(init_missing, int):
            numpy.random.seed(init_missing)
        elif init_missing != 'zero':
            raise ValueError(f"invalid {init_missing=}")

        if len(set(alphabet)) != len(alphabet):
            raise ValueError('duplicate letters in `alphabet`')
        self.alphabet = tuple(alphabet)
        chars = []
        for char in self.alphabet:
            if char.isalpha():
                chars.append(char)
            elif char == '*':
                chars.append(r'\*')
            else:
                raise ValueError(f"invalid alphabet character: {char}")
        chars = '|'.join(chars)
        self._mutation_regex = re.compile(rf"(?P<wt>{chars})"
                                          rf"(?P<site>\d+)"
                                          rf"(?P<mut>{chars})")

        if (activity_wt_df is not None) and (mut_escape_df is not None):
            if n_epitopes is not None:
                raise ValueError('specify `activity_wt_df` or `n_epitopes`')

            if pd.isnull(activity_wt_df['epitope']).any():
                raise ValueError('epitope name cannot be null')
            self.epitopes = tuple(activity_wt_df['epitope'].unique())
            if len(self.epitopes) != len(activity_wt_df):
                raise ValueError('duplicate epitopes in `activity_wt_df`')

        elif (activity_wt_df is None) and (mut_escape_df is None):
            if not isinstance(n_epitopes, int) and n_epitopes > 0:
                raise ValueError('`n_epitopes` must be int > 1 if no '
                                 '`activity_wt_df`')
            self.epitopes = tuple(f"epitope {i + 1}" for
                                  i in range(n_epitopes))

            # initialize activities
            activity_wt_df = pd.DataFrame({
                    'epitope': self.epitopes,
                    'activity': (0.0 if init_missing == 'zero' else
                                 numpy.random.rand(len(self.epitopes)))
                    })

            if data_to_fit is None:
                raise ValueError('specify `data_to_fit` if `activity_wt_df` '
                                 'and `mut_escape_df` are `None`')

        else:
            raise ValueError('initialize both or neither `activity_wt_df` '
                             'and `mut_escape_df`')

        if isinstance(epitope_colors, dict):
            self.epitope_colors = {epitope_colors[e] for e in self.epitopes}
        elif len(epitope_colors) < len(self.epitopes):
            raise ValueError('not enough `epitope_colors`')
        else:
            self.epitope_colors = dict(zip(self.epitopes, epitope_colors))

        def _init_mut_escape_df(mutations):
            # initialize mutation escape values
            if init_missing == 'zero':
                init = 0.0
            else:
                init = numpy.random.rand(len(self.epitopes) * len(mutations))
            return pd.DataFrame(
                    {'epitope': list(self.epitopes) * len(mutations),
                     'mutation': [m for m in mutations for _ in self.epitopes],
                     'escape': init
                     })

        # get wildtype, sites, and mutations
        if mut_escape_df is not None:
            wts, sites, muts = self._muts_from_mut_escape_df(mut_escape_df)
        if data_to_fit is not None:
            wts2, sites2, muts2 = self._muts_from_data_to_fit(data_to_fit)
        if mut_escape_df is data_to_fit is None:
            raise ValueError('initialize `mut_escape_df` or `data_to_fit`')
        elif mut_escape_df is None:
            self.wts, self.sites, self.mutations = wts2, sites2, muts2
            mut_escape_df = _init_mut_escape_df(self.mutations)
        elif data_to_fit is None:
            self.wts, self.sites, self.mutations = wts, sites, muts
        else:
            if data_mut_escape_overlap == 'exact_match':
                if sites == sites2 and wts == wts2 and muts == muts2:
                    self.wts, self.sites, self.mutations = wts, sites, muts
                else:
                    raise ValueError('`data_to_fit` and `mut_escape_df` give '
                                     'different mutations. Fix or set '
                                     'data_mut_escape_overlap="fill_to_data"')
            elif data_mut_escape_overlap == 'fill_to_data':
                if set(sites) < set(sites2):
                    self.sites = sites2
                else:
                    raise ValueError('`mut_escape_df` has more sites than '
                                     '`data_to_fit`')
                if wts.items() < wts2.items():
                    self.wts = wts2
                else:
                    raise ValueError('`mut_escape_df` has wts not in '
                                     '`data_to_fit`')
                if set(muts) < set(muts2):
                    self.mutations = muts2
                else:
                    raise ValueError('`mut_escape_df` has mutations not in '
                                     '`data_to_fit`')
                # take values from `mut_escape_df` and fill missing
                mut_escape_df = (
                    mut_escape_df
                    .set_index(['epitope', 'mutation'])
                    ['escape']
                    .combine_first(_init_mut_escape_df(self.mutations)
                                   .set_index(['epitope', 'mutation'])
                                   ['escape'])
                    .reset_index()
                    )
            else:
                raise ValueError(f"invalid {data_mut_escape_overlap=}")

        if set(mut_escape_df['epitope']) != set(self.epitopes):
            raise ValueError('`mut_escape_df` does not have same epitopes as '
                             '`activity_wt_df`')
        for epitope, df in mut_escape_df.groupby('epitope'):
            if sorted(df['mutation']) != sorted(self.mutations):
                raise ValueError(f"invalid set of mutations for {epitope=}")

        # set internal params with activities and escapes
        self._params = self._params_from_dfs(activity_wt_df, mut_escape_df)

        if data_to_fit is not None:
            (self._one_binarymap, self._binarymaps,
             self._cs, self._pvs, self.data_to_fit
             ) = self._binarymaps_cs_pvs_from_df(data_to_fit, get_pv=True)
            assert len(self._pvs) == len(self.data_to_fit)
        else:
            self.data_to_fit = None

    def _binarymaps_cs_pvs_from_df(self, df, get_pv):
        """Get variants and concentrations from data frame.

        Get `(one_binarymap, binarymaps, cs, pvs, sorted_df)`. If
        `get_pv=False` then `pvs` is `None`. If same variants for all
        concentrations, `binarymaps` is a BinaryMap and `one_binarymap` is
        `True`. Otherwise, `binarymaps` lists BinaryMap for each concentration.
        `sorted_df` is version of `df` with variants/concentrations in same
        order as `binarymaps`, while `pvs` is 1D array with the prob escapes
        for these variants. We handle separately cases when BinaryMap same or
        different for concentrations as more efficient if all concentrations
        have same BinaryMap.

        """
        cols = ['concentration', 'aa_substitutions']
        if get_pv:
            cols.append('prob_escape')
        if not df[cols].notnull().all().all():
            raise ValueError(f"null entries in data frame of variants:\n{df}")
        sorted_df = (df
                     .sort_values(['concentration', 'aa_substitutions'])
                     .reset_index(drop=True)
                     )
        cs = (sorted_df
              ['concentration']
              .astype(float)
              .sort_values()
              .unique()
              )
        if not (cs > 0).all():
            raise ValueError('concentrations must be > 0')
        binarymaps = []
        pvs = [] if get_pv else None
        one_binarymap = True
        for i, (c, i_df) in enumerate(sorted_df.groupby('concentration',
                                                        sort=False)):
            assert c == cs[i]
            i_variants = i_df['aa_substitutions'].reset_index(drop=True)
            if i == 0:
                first_variants = i_variants
            elif one_binarymap:
                one_binarymap = first_variants.equals(i_variants)
            binarymaps.append(self._get_binarymap(i_df))
            if get_pv:
                pvs.append(i_df['prob_escape'].to_numpy(dtype=float))
        if one_binarymap:
            binarymaps = binarymaps[0]
            if get_pv:
                assert all(binarymaps.nvariants == len(pv) for pv in pvs)
                pvs = numpy.concatenate(pvs)
                assert len(pvs) == len(sorted_df)
                if (pvs < 0).any() or (pvs > 1).any():
                    raise ValueError('`prob_escape` must be between 0 and 1')
        return (one_binarymap, binarymaps, cs, pvs, sorted_df)

    def _params_from_dfs(self, activity_wt_df, mut_escape_df):
        """Params vector from data frames of activities and escapes."""
        # first E entries are activities
        assert len(activity_wt_df) == len(self.epitopes)
        assert len(self.epitopes) == activity_wt_df['epitope'].nunique()
        assert set(self.epitopes) == set(activity_wt_df['epitope'])
        params = (
            activity_wt_df
            .assign(epitope=lambda x: pd.Categorical(x['epitope'],
                                                     self.epitopes,
                                                     ordered=True)
                    )
            .sort_values('epitope')
            ['activity']
            .tolist()
            )

        # Remaining MxE entries are beta values
        assert len(mut_escape_df) == len(self.epitopes) * len(self.mutations)
        assert len(mut_escape_df) == len(mut_escape_df
                                         .groupby(['mutation', 'epitope']))
        assert set(self.epitopes) == set(mut_escape_df['epitope'])
        assert set(self.mutations) == set(mut_escape_df['mutation'])
        params.extend(
            mut_escape_df
            .assign(epitope=lambda x: pd.Categorical(x['epitope'],
                                                     self.epitopes,
                                                     ordered=True),
                    mutation=lambda x: pd.Categorical(x['mutation'],
                                                      self.mutations,
                                                      ordered=True),
                    )
            .sort_values(['mutation', 'epitope'])
            ['escape']
            .tolist()
            )

        params = numpy.array(params).astype(float)
        if numpy.isnan(params).any():
            raise ValueError('some parameters are NaN')
        return params

    def _a_beta_from_params(self, params):
        """Vector of activities and MxE matrix of betas from params vector."""
        params_len = len(self.epitopes) * (1 + len(self.mutations))
        if params.shape != (params_len,):
            raise ValueError(f"invalid {params.shape=}")
        a = params[: len(self.epitopes)]
        beta = params[len(self.epitopes):].reshape(len(self.mutations),
                                                   len(self.epitopes))
        assert a.shape == (len(self.epitopes),)
        assert beta.shape == (len(self.mutations), len(self.epitopes))
        assert (not numpy.isnan(a).any()) and (not numpy.isnan(beta).any())
        return (a, beta)

    def _muts_from_data_to_fit(self, data_to_fit):
        """Get wildtypes, sites, and mutations from ``data_to_fit``."""
        wts = {}
        mutations = collections.defaultdict(set)
        for variant in data_to_fit['aa_substitutions']:
            for mutation in variant.split():
                wt, site, _ = self._parse_mutation(mutation)
                if site not in wts:
                    wts[site] = wt
                elif wts[site] != wt:
                    raise ValueError(f"inconsistent wildtype for site {site}")
                mutations[site].add(mutation)
        sites = tuple(sorted(wts.keys()))
        wts = dict(sorted(wts.items()))
        assert set(mutations.keys()) == set(sites) == set(wts)
        char_order = {c: i for i, c in enumerate(self.alphabet)}
        mutations = tuple(mut for site in sites for mut in
                          sorted(mutations[site],
                                 key=lambda m: char_order[m[-1]]))
        return (wts, sites, mutations)

    def _muts_from_mut_escape_df(self, mut_escape_df):
        """Get wildtypes, sites, and mutations from ``mut_escape_df``."""
        wts = {}
        mutations = collections.defaultdict(set)
        for mutation in mut_escape_df['mutation'].unique():
            wt, site, _ = self._parse_mutation(mutation)
            if site not in wts:
                wts[site] = wt
            elif wts[site] != wt:
                raise ValueError(f"inconsistent wildtype for site {site}")
            mutations[site].add(mutation)
        sites = tuple(sorted(wts.keys()))
        wts = dict(sorted(wts.items()))
        assert set(mutations.keys()) == set(sites) == set(wts)
        char_order = {c: i for i, c in enumerate(self.alphabet)}
        mutations = tuple(mut for site in sites for mut in
                          sorted(mutations[site],
                                 key=lambda m: char_order[m[-1]]))
        return (wts, sites, mutations)

    @property
    def activity_wt_df(self):
        r"""pandas.DataFrame: Activities :math:`a_{\rm{wt,e}}` for epitopes."""
        a, _ = self._a_beta_from_params(self._params)
        assert a.shape == (len(self.epitopes),)
        return pd.DataFrame({'epitope': self.epitopes,
                             'activity': a,
                             })

    @property
    def mut_escape_df(self):
        r"""pandas.DataFrame: Escape :math:`\beta_{m,e}` for each mutation."""
        _, beta = self._a_beta_from_params(self._params)
        assert beta.shape == (len(self.mutations), len(self.epitopes))
        return (pd.concat([pd.DataFrame({'mutation': self.mutations,
                                         'escape': b,
                                         'epitope': e,
                                         })
                           for e, b in zip(self.epitopes, beta.transpose())],
                          ignore_index=True)
                .assign(
                    site=lambda x: x['mutation'].map(
                                        lambda m: self._parse_mutation(m)[1]),
                    mutant=lambda x: x['mutation'].map(
                                        lambda m: self._parse_mutation(m)[2]),
                    wildtype=lambda x: x['site'].map(self.wts),
                    )
                [['epitope', 'site', 'wildtype', 'mutant',
                  'mutation', 'escape']]
                )

    @property
    def mut_escape_site_summary_df(self):
        r"""pandas.DataFrame: Site-level summaries of mutation escape."""
        escape_metrics = {
                'mean': pd.NamedAgg('escape', 'mean'),
                'total positive': pd.NamedAgg('escape_gt_0', 'sum'),
                'max': pd.NamedAgg('escape', 'max'),
                'min': pd.NamedAgg('escape', 'min'),
                'total negative': pd.NamedAgg('escape_lt_0', 'sum'),
                }
        return (
            self.mut_escape_df
            .assign(escape_gt_0=lambda x: x['escape'].clip(lower=0),
                    escape_lt_0=lambda x: x['escape'].clip(upper=0),
                    )
            .groupby(['epitope', 'site', 'wildtype'],
                     as_index=False,
                     sort=False)
            .aggregate(**escape_metrics)
            )

    def prob_escape(self,
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
        prob_escape_col = 'predicted_prob_escape'
        if prob_escape_col in variants_df.columns:
            raise ValueError(f"`variants_df` has column {prob_escape_col}")

        # add concentrations column to variants_df
        if concentrations is not None:
            if 'concentration' in variants_df.columns:
                raise ValueError('`variants_df` has "concentration" column '
                                 'and `concentrations` not `None`')
            variants_df = pd.concat([variants_df.assign(concentration=c)
                                     for c in concentrations],
                                    ignore_index=True)

        (one_binarymap, binarymaps, cs, _, variants_df
         ) = self._binarymaps_cs_pvs_from_df(variants_df, get_pv=False)

        p_v_c = self._compute_1d_pvs(self._params, one_binarymap,
                                     binarymaps, cs)
        assert p_v_c.shape == (len(variants_df),)
        variants_df[prob_escape_col] = p_v_c

        return variants_df

    def fit(self,
            *,
            loss_type='L1',
            method='scipy_minimize',
            scipy_solver='L-BFGS-B',
            ):
        r"""Fit parameters (activities and mutation escapes) to the data.

        Requires :attr:`Polyclonal.data_to_fit` be set at initialization of
        this :class:`Polyclonal` object. After calling this method, the
        :math:`a_{\rm{wt},e}` and :math:`\beta_{m,e}` have been optimized, and
        can be accessed using other methods of the :class:`Polyclonal` object.

        Parameters
        ----------
        loss_type : {'L1', 'L2'}
            Minimize difference between actual and model-predicted
            :math:`p_v\left(c\right)` using L1 or L2 loss function.
        method : {'scipy_minimize'}
            Approach used for fitting.
        scipy_solver : str
            If ``method='scipy_minimize'``, what solver to use.

        Return
        ------
        scipy.optimize.OptimizeResult
            Return value depends on ``method``.

        """
        if self.data_to_fit is None:
            raise ValueError('cannot fit if `data_to_fit` not set')

        def _loss_func(params):
            pred_pvs = self._compute_1d_pvs(params, self._one_binarymap,
                                            self._binarymaps, self._cs)
            assert pred_pvs.shape == self._pvs.shape
            if loss_type == 'L1':
                loss = numpy.absolute(self._pvs - pred_pvs).sum()
            elif loss_type == 'L2':
                loss = numpy.sum((self._pvs - pred_pvs)**2)
            else:
                raise ValueError(f"invalid {loss_type=}")
            return loss

        if method == 'scipy_minimize':
            opt_res = scipy.optimize.minimize(fun=_loss_func,
                                              x0=self._params,
                                              method=scipy_solver,
                                              )
            self._params = opt_res.x
            if not opt_res.success:
                raise RuntimeError(f"optimization failed:\n{opt_res}")
            return opt_res

        else:
            raise ValueError(f"invalid {method=}")

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
        kwargs['activity_wt_df'] = self.activity_wt_df
        if 'epitope_colors' not in kwargs:
            kwargs['epitope_colors'] = self.epitope_colors
        return polyclonal.plot.activity_wt_barplot(**kwargs)

    def mut_escape_pdb_b_factor(self,
                                *,
                                input_pdbfile,
                                chains,
                                metric,
                                outdir=None,
                                outfile='{metric}-{epitope}.pdb',
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
        if (metric in df.columns) and (metric not in
                                       {'epitope', 'site', 'wildtype'}):
            metric_col = metric
        if isinstance(chains, str) and len(chains) == 1:
            chains = [chains]
        df = pd.concat([df.assign(chain=chain) for chain in chains],
                       ignore_index=True)
        result_files = []
        for epitope in self.epitopes:
            if outdir:
                output_pdbfile = os.path.join(outdir, outfile)
            else:
                output_pdbfile = outfile
            output_pdbfile = output_pdbfile.format(epitope=epitope,
                                                   metric=metric
                                                   ).replace(' ', '_')
            if os.path.dirname(output_pdbfile):
                os.makedirs(os.path.dirname(output_pdbfile), exist_ok=True)
            result_files.append((epitope, output_pdbfile))
            polyclonal.pdb_utils.reassign_b_factor(
                        input_pdbfile,
                        output_pdbfile,
                        df.query('epitope == @epitope'),
                        metric_col,
                        missing_metric=missing_metric,
                        )
        return pd.DataFrame(result_files, columns=['epitope', 'PDB file'])

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
        kwargs['mut_escape_site_summary_df'] = self.mut_escape_site_summary_df
        if 'epitope_colors' not in kwargs:
            kwargs['epitope_colors'] = self.epitope_colors
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
        kwargs['mut_escape_df'] = self.mut_escape_df
        if 'epitope_colors' not in kwargs:
            kwargs['epitope_colors'] = self.epitope_colors
        if 'alphabet' not in kwargs:
            kwargs['alphabet'] = self.alphabet
        return polyclonal.plot.mut_escape_heatmap(**kwargs)

    def _compute_1d_pvs(self, params, one_binarymap, binarymaps, cs):
        r"""Get 1D raveled array of :math:`p_v\left(c\right)` values.

        Differs from :meth:`Polyclonal._compute_pv` in that it works if just
        one or multiple BinaryMap objects.

        """
        if one_binarymap:
            p_v_c = self._compute_pv(params, binarymaps, cs)
            assert p_v_c.shape == (binarymaps.nvariants, len(cs))
            return p_v_c.ravel(order='F')
        else:
            assert len(cs) == len(binarymaps)
            return numpy.concatenate(
                    [self._compute_pv(params, bmap, numpy.array([c])).ravel()
                     for c, bmap in zip(cs, binarymaps)])

    def _compute_pv(self, params, bmap, cs):
        r"""Compute :math:`p_v\left(c\right)`.

        Takes set of params, a single BinaryMap, and array of concentrations,
        and returns nvariants X nconcentrations array of the p_v values.

        """
        a, beta = self._a_beta_from_params(params)
        assert a.shape == (len(self.epitopes),)
        assert beta.shape == (bmap.binarylength, len(self.epitopes))
        assert beta.shape[0] == bmap.binary_variants.shape[1]
        assert (cs > 0).all()
        assert cs.ndim == 1
        phi_e_v = bmap.binary_variants.dot(beta) - a
        assert phi_e_v.shape == (bmap.nvariants, len(self.epitopes))
        exp_minus_phi_e_v = numpy.exp(-phi_e_v)
        U_e_v_c = 1.0 / (1.0 + numpy.multiply.outer(exp_minus_phi_e_v, cs))
        assert U_e_v_c.shape == (bmap.nvariants, len(self.epitopes), len(cs))
        p_v_c = U_e_v_c.prod(axis=1)
        assert p_v_c.shape == (bmap.nvariants, len(cs))
        return p_v_c

    def _get_binarymap(self,
                       variants_df,
                       ):
        """Get ``BinaryMap`` appropriate for use."""
        bmap = binarymap.BinaryMap(
                variants_df,
                substitutions_col='aa_substitutions',
                allowed_subs=self.mutations,
                )
        assert tuple(bmap.all_subs) == self.mutations
        return bmap

    def _parse_mutation(self, mutation):
        """Return `(wt, site, mut)`."""
        m = self._mutation_regex.fullmatch(mutation)
        if not m:
            raise ValueError(f"invalid mutation {mutation}")
        else:
            return (m.group('wt'), int(m.group('site')), m.group('mut'))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
