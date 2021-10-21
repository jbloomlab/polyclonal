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
    ValueError: not all expected mutations for e2

    Now make a data frame with some variants:

    >>> variants_df = pd.DataFrame.from_records(
    ...         [('AA', 'A2K'),
    ...          ('AC', 'M1A A2K'),
    ...          ('AG', 'M1A'),
    ...          ('AT', ''),
    ...          ('CA', 'A2K')],
    ...         columns=['barcode', 'aa_substitutions'])

    Get the escape probabilities:

    >>> polyclonal.prob_escape(variants_df=variants_df,
    ...                        concentrations=[1, 2, 4]).round(3)
       barcode aa_substitutions  concentration  prob_escape
    0       AA              A2K            1.0        0.097
    1       AC          M1A A2K            1.0        0.598
    2       AG              M1A            1.0        0.197
    3       AT                             1.0        0.032
    4       CA              A2K            1.0        0.097
    5       AA              A2K            2.0        0.044
    6       AC          M1A A2K            2.0        0.398
    7       AG              M1A            2.0        0.090
    8       AT                             2.0        0.010
    9       CA              A2K            2.0        0.044
    10      AA              A2K            4.0        0.017
    11      AC          M1A A2K            4.0        0.214
    12      AG              M1A            4.0        0.034
    13      AT                             4.0        0.003
    14      CA              A2K            4.0        0.017

    """

    def __init__(self,
                 *,
                 activity_wt_df=None,
                 mut_escape_df=None,
                 data_to_fit=None,
                 n_epitopes=None,
                 alphabet=AAS_NOSTOP,
                 epitope_colors=polyclonal.plot.TAB10_COLORS_NOGRAY,
                 ):
        """See main class docstring."""
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
            self._activity_wt = (activity_wt_df
                                 .set_index('epitope')
                                 ['activity']
                                 .astype(float)
                                 .to_dict()
                                 )

        elif (activity_wt_df is None) and (mut_escape_df is None):
            if not isinstance(n_epitopes, int) and n_epitopes > 0:
                raise ValueError('`n_epitopes` must be int > 1 if no '
                                 '`activity_wt_df`')
            epitopes = tuple(f"epitope {i + 1}" for i in range(n_epitopes))

            # initialize activities to all be zero
            self._activity_wt = {epitope: 0.0 for epitope in epitopes}

            if data_to_fit is None:
                raise ValueError('specify `data_to_fit` or `mut_escape_df`')

            raise NotImplementedError('not yet set up to handle `data_to_fit`')

        else:
            raise ValueError('initialize both or neither `activity_wt_df` '
                             'and `mut_escape_df`')

        if data_to_fit is not None:
            raise NotImplementedError('not yet set up to handle `data_to_fit`')

        if isinstance(epitope_colors, dict):
            self.epitope_colors = {epitope_colors[e] for e in self.epitopes}
        elif len(epitope_colors) < len(self.epitopes):
            raise ValueError('not enough `epitope_colors`')
        else:
            self.epitope_colors = dict(zip(self.epitopes, epitope_colors))

        # get wildtype, sites, and mutations
        if mut_escape_df is not None:
            self.wts, self.sites, mutations = (
                    self._mutations_from_mut_escape_df(mut_escape_df))
        if data_to_fit is not None:
            raise NotImplementedError
        assert set(mutations.keys()) == set(self.sites) == set(self.wts)
        char_order = {c: i for i, c in enumerate(self.alphabet)}
        self.mutations = tuple(mut for site in self.sites for mut in
                               sorted(mutations[site],
                                      key=lambda m: char_order[m[-1]]))

        # get mutation escape values
        if set(mut_escape_df['epitope']) != set(self.epitopes):
            raise ValueError('`mut_escape_df` does not have same epitopes as '
                             '`activity_wt_df`')
        self._mut_escape = {}
        for epitope, df in mut_escape_df.groupby('epitope'):
            if set(df['mutation']) != set(self.mutations):
                raise ValueError(f"not all expected mutations for {epitope}")
            self._mut_escape[epitope] = (df
                                         .set_index('mutation')
                                         ['escape']
                                         .astype(float)
                                         .to_dict()
                                         )
        assert set(self.epitopes) == set(self._activity_wt)
        assert set(self.epitopes) == set(self._mut_escape)

        # below are set to non-null values in `_set_binarymap` when
        # specific variants provided
        self._binarymap = None
        self._beta = None  # M by E matrix of betas
        self._a = None  # length E vector of activities

    def _mutations_from_mut_escape_df(self, mut_escape_df):
        """Get wildtypes, sites, and mutations from ``mut_escape_df``.

        Parameters
        ----------
        mut_escape_df : pandas.DataFrame

        Returns
        -------
        (wts, sites, mutations)

        """
        wts = {}
        mutations = collections.defaultdict(list)
        for mutation in mut_escape_df['mutation'].unique():
            wt, site, mut = self._parse_mutation(mutation)
            if site not in wts:
                wts[site] = wt
            elif wts[site] != wt:
                raise ValueError(f"inconsistent wildtype for site {site}")
            mutations[site].append(mutation)
        sites = tuple(sorted(wts.keys()))
        wts = dict(sorted(wts.items()))
        return (wts, sites, mutations)

    @property
    def activity_wt_df(self):
        r"""pandas.DataFrame: Activities :math:`a_{\rm{wt,e}}` for epitopes."""
        return pd.DataFrame({'epitope': self.epitopes,
                             'activity': [self._activity_wt[e]
                                          for e in self.epitopes],
                             })

    @property
    def mut_escape_df(self):
        r"""pandas.DataFrame: Escape :math:`\beta_{m,e}` for each mutation."""
        return (pd.concat([pd.DataFrame({'mutation': self.mutations,
                                         'escape': [self._mut_escape[e][m]
                                                    for m in self.mutations],
                                         })
                           .assign(epitope=e)
                           for e in self.epitopes],
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
                    concentrations,
                    ):
        r"""Compute probability of escape :math:`p_v\left(c\right)`.

        Arguments
        ---------
        variants_df : pandas.DataFrame
            Input data frame defining variants. Should have a column
            named 'aa_substitutions' that definese variants as space-delimited
            strings of substitutions (e.g., 'M1A K3T').
        concentrations : array-like
            Concentrations at which we compute probability of escape.

        Returns
        -------
        pandas.DataFrame
            A copy of ``variants_df`` with new columns named 'concentration'
            and 'prob_escape' giving probability of escape
            :math:`p_v\left(c\right)` for each variant at each concentration.

        """
        concentration_col = 'concentration'
        prob_escape_col = 'prob_escape'
        for col in [concentration_col, prob_escape_col]:
            if col in variants_df.columns:
                raise ValueError(f"`variants_df` already has column {col}")
        self._set_binarymap(variants_df, substitutions_col='aa_substitutions')
        cs = numpy.array(concentrations, dtype='float')
        if not (cs > 0).all():
            raise ValueError('concentrations must be > 0')
        if cs.ndim != 1:
            raise ValueError('concentrations must be 1-dimensional')
        p_v_c = self._compute_pv(cs)
        assert p_v_c.shape == (self._binarymap.nvariants, len(cs))
        return (pd.concat([variants_df.assign(**{concentration_col: c})
                           for c in cs],
                          ignore_index=True)
                .assign(**{prob_escape_col: p_v_c.ravel(order='F')})
                )

    def fit(self):
        """Not yet implemented."""
        raise NotImplementedError

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

    def _compute_pv(self, cs):
        r"""Compute :math:`p_v\left(c\right)`. Call `_set_binarymap` first."""
        if self._binarymap is None or self._a is None or self._beta is None:
            raise ValueError('call `_set_binarymap` first')
        assert (cs > 0).all()
        assert cs.ndim == 1
        phi_e_v = self._binarymap.binary_variants.dot(self._beta) - self._a
        assert phi_e_v.shape == (self._binarymap.nvariants, len(self.epitopes))
        exp_minus_phi_e_v = numpy.exp(-phi_e_v)
        U_e_v_c = 1.0 / (1.0 + numpy.multiply.outer(exp_minus_phi_e_v, cs))
        assert U_e_v_c.shape == (self._binarymap.nvariants,
                                 len(self.epitopes),
                                 len(cs))
        p_v_c = U_e_v_c.prod(axis=1)
        assert p_v_c.shape == (self._binarymap.nvariants, len(cs))
        return p_v_c

    def _set_binarymap(self,
                       variants_df,
                       substitutions_col,
                       ):
        """Set `_binarymap`, `_beta`, `_a` attributes."""
        self._binarymap = binarymap.BinaryMap(
                variants_df,
                substitutions_col=substitutions_col,
                )
        extra_muts = set(self._binarymap.all_subs) - set(self.mutations)
        if extra_muts:
            raise ValueError('variants contain mutations for which no '
                             'escape value initialized:\n'
                             '\n'.join(extra_muts))

        self._a = numpy.array([self._activity_wt[e] for e in self.epitopes],
                              dtype='float')
        assert self._a.shape == (len(self.epitopes),)

        self._beta = numpy.array(
                        [[self._mut_escape[e][m] for e in self.epitopes]
                         for m in self._binarymap.all_subs],
                        dtype='float')
        assert self._beta.shape == (self._binarymap.binarylength,
                                    len(self.epitopes))
        assert self._beta.shape[0] == self._binarymap.binary_variants.shape[1]

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
