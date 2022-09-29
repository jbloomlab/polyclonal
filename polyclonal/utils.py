"""
===========
utils
===========

Miscellaneous utility functions.

"""


import re

import pandas as pd  # noqa: F401

import polyclonal


class MutationParser:
    """Parse mutation strings like 'A5G'.

    Parameters
    ----------
    alphabet : array-like
        Valid single-character letters in alphabet.
    letter_suffixed_sites : bool
        Allow sites suffixed by lowercase letters, such as "214a". In this case, returned
        sites from :meth:`MutationParser.parse_mut` are str.

    Example
    -------
    >>> mutparser = MutationParser(polyclonal.AAS)
    >>> mutparser.parse_mut('A5G')
    ('A', 5, 'G')

    >>> mutparser.parse_mut('K7-')
    Traceback (most recent call last):
      ...
    ValueError: invalid mutation K7-

    >>> mutparser_gap = MutationParser(polyclonal.AAS_WITHGAP)
    >>> mutparser_gap.parse_mut('K7-')
    ('K', 7, '-')

    >>> mutparser.parse_mut("E214aA")
    Traceback (most recent call last):
      ...
    ValueError: invalid mutation E214aA

    >>> mutparser_letter_suffix = MutationParser(polyclonal.AAS, True)
    >>> mutparser_letter_suffix.parse_mut('A5G')
    ('A', '5', 'G')
    >>> mutparser_letter_suffix.parse_mut('E214aA')
    ('E', '214a', 'A')

    >>> mutparser.parse_mut("A-1G")
    ('A', -1, 'G')

    """

    def __init__(self, alphabet, letter_suffixed_sites=False):
        """See main class docstring."""
        chars = []
        for char in alphabet:
            if char.isalpha():
                chars.append(char)
            elif char == "*":
                chars.append(r"\*")
            elif char == "-":
                chars.append(r"\-")
            else:
                raise ValueError(f"invalid alphabet character: {char}")
        chars = "|".join(chars)
        if letter_suffixed_sites:
            self._sites_as_int = False
            site_regex = r"(?P<site>\-?\d+[a-z]?)"
        else:
            self._sites_as_int = True
            site_regex = r"(?P<site>\-?\d+)"
        self._mutation_regex = re.compile(
            rf"(?P<wt>{chars})" + site_regex + rf"(?P<mut>{chars})"
        )

    def parse_mut(self, mutation):
        """tuple: `(wildtype, site, mutation)`."""
        m = self._mutation_regex.fullmatch(mutation)
        if not m:
            raise ValueError(f"invalid mutation {mutation}")
        else:
            site = int(m.group("site")) if self._sites_as_int else m.group("site")
            return (m.group("wt"), site, m.group("mut"))


def site_level_variants(
    df,
    *,
    original_alphabet=polyclonal.AAS,
    wt_char="w",
    mut_char="m",
    letter_suffixed_sites=False,
):
    """Re-define variants simply in terms of which sites are mutated.

    This function is useful if you have a data frame of variants and you
    want to simplify them from full mutations to just indicating whether
    sites are mutated.

    Parameters
    ----------
    df : pandas.DataFrame
        Must include a column named 'aa_substitutions'.
    original_alphabet : array-like
        Valid single-letter characters in the original (mutation-level)
        alphabet.
    wt_char : str
        Single letter used to represent wildtype identity at all sites.
    mut_char : str
        Single letter used to represent mutant identity at all sites.
    letter_suffixed_sites : str
        Same mutation as for :class:`MutationParser`.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with 'aa_substitutions' in site-level encoding.

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame.from_records(
    ...         [('AA', 'M1A', 1.0),
    ...          ('AC', '', 0.0),
    ...          ('AG', 'M1A C53T', 1.0),
    ...          ],
    ...        columns=['barcode', 'aa_substitutions', 'escape'],
    ...        )
    >>> site_level_variants(df)
      barcode aa_substitutions  escape
    0      AA              w1m     1.0
    1      AC                      0.0
    2      AG         w1m w53m     1.0

    """
    subs_col = "aa_substitutions"
    if subs_col not in df.columns:
        raise ValueError(f"`df` lacks column {subs_col}")
    if isinstance(wt_char, str) and len(wt_char) != 1:
        raise ValueError(f"{wt_char=} should be single letter")
    if isinstance(mut_char, str) and len(mut_char) != 1:
        raise ValueError(f"{mut_char=} should be single letter")
    mutparser = MutationParser(
        original_alphabet,
        letter_suffixed_sites=letter_suffixed_sites,
    )

    site_subs_mapping = {}
    wts = {}
    for subs in df[subs_col]:
        if subs not in site_subs_mapping:
            site_subs = []
            for sub in subs.split():
                wt, site, _ = mutparser.parse_mut(sub)
                if site in wts and wts[site] != wt:
                    raise ValueError(
                        f"inconsistent wildtype at {site}: " f"{wt} versus {wts[site]}"
                    )
                else:
                    wts[site] = wt
                site_subs.append(f"{wt_char}{site}{mut_char}")
            site_subs_mapping[subs] = " ".join(site_subs)

    df = df.copy()
    df[subs_col] = df[subs_col].map(site_subs_mapping)
    return df


def shift_mut_site(mut_str, shift):
    """Shift site in string of mutations.

    Parameters
    ----------
    mut_str : str
        String of space-delimited amino-acid substitution mutations.
    shift : int
        Amount to shift sites (add this to current site number).

    Returns
    -------
    str
        Mutation string with sites shifted.

    Example
    -------
    >>> shift_mut_site('A1G K7A', 2)
    'A3G K9A'

    """
    if not isinstance(shift, int):
        raise ValueError("shift must be int")
    new_mut_str = []
    for mut in mut_str.split():
        m = re.fullmatch(r"(?P<wt>\S)(?P<site>\d+)(?P<mut>\S)", mut)
        if not m:
            raise ValueError(f"cannot match {mut} in {mut_str}")
        new_site = int(m.group("site")) + shift
        new_mut_str.append(f"{m.group('wt')}{new_site}{m.group('mut')}")
    return " ".join(new_mut_str)


def tidy_to_corr(
    df,
    sample_col,
    label_col,
    value_col,
    *,
    group_cols=None,
    return_type="tidy_pairs",
    method="pearson",
):
    """Pairwise correlations between samples in tidy data frame.

    Parameters
    ----------
    df : pandas.DataFrame
        Tidy data frame.
    sample_col : str
        Column in `df` with name of sample.
    label_col : str
        Column in `df` with labels for variable to correlate.
    value_col : str
        Column in `df` with values to correlate.
    group_cols : None, str, or list
        Additional columns used to group results.
    return_type : {'tidy_pairs', 'matrix'}
        Return results as tidy dataframe of pairwise correlations
        or correlation matrix.
    method : str
        A correlation method passable to `pandas.DataFrame.corr`.

    Returns
    -------
    pandas.DataFrame
        Holds pairwise correlations in format specified by `return_type`.
        Correlations only calculated among values with shared label
        among samples.

    Example
    -------
    Define data frame with data to correlate:

    >>> df = pd.DataFrame({
    ...        'sample': ['a', 'a', 'a', 'b', 'b', 'b', 'b', 'c', 'c', 'c'],
    ...        'barcode': ['A', 'C', 'G', 'G', 'A', 'C', 'T', 'G', 'C', 'A'],
    ...        'score': [1, 2, 3, 3, 1.5, 2, 4, 1, 2, 3],
    ...        'group': ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'y', 'y', 'y'],
    ...        })

    Pairwise correlations between all samples ignoring group:

    >>> tidy_to_corr(df, sample_col='sample', label_col='barcode',
    ...              value_col='score')
      sample_1 sample_2  correlation
    0        a        a     1.000000
    1        b        a     0.981981
    2        c        a    -1.000000
    3        a        b     0.981981
    4        b        b     1.000000
    5        c        b    -0.981981
    6        a        c    -1.000000
    7        b        c    -0.981981
    8        c        c     1.000000

    The same but as a matrix rather than in tidy format:

    >>> tidy_to_corr(df, sample_col='sample', label_col='barcode',
    ...              value_col='score', return_type='matrix')
      sample         a         b         c
    0      a  1.000000  0.981981 -1.000000
    1      b  0.981981  1.000000 -0.981981
    2      c -1.000000 -0.981981  1.000000

    Now group before computing correlations:

    >>> tidy_to_corr(df, sample_col='sample', label_col='barcode',
    ...              value_col='score', group_cols='group')
      group sample_1 sample_2  correlation
    0     x        a        a     1.000000
    1     x        b        a     0.981981
    2     x        a        b     0.981981
    3     x        b        b     1.000000
    4     y        c        c     1.000000
    >>> tidy_to_corr(df, sample_col='sample', label_col='barcode',
    ...              value_col='score', group_cols='group',
    ...              return_type='matrix')
      group sample         a         b    c
    0     x      a  1.000000  0.981981  NaN
    1     x      b  0.981981  1.000000  NaN
    2     y      c       NaN       NaN  1.0

    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    elif group_cols is None:
        group_cols = []
    cols = [sample_col, value_col, label_col] + group_cols
    if set(cols) > set(df.columns):
        raise ValueError(f"`df` missing some of these columns: {cols}")
    if len(set(cols)) != len(cols):
        raise ValueError(f"duplicate column names: {cols}")
    if "correlation" in cols:
        raise ValueError("cannot have column named `correlation`")
    if sample_col + "_2" in group_cols:
        raise ValueError(f"cannot have column named `{sample_col}_2`")

    for _, g in df.groupby(
        [sample_col, *group_cols] if len(group_cols) else sample_col
    ):
        if len(g[label_col]) != g[label_col].nunique():
            raise ValueError(
                f"Entries in `df` column {label_col} not unique "
                "after grouping by: " + ", ".join(c for c in [sample_col] + group_cols)
            )

    df = df.pivot_table(
        values=value_col,
        columns=sample_col,
        index=[label_col] + group_cols,
    ).reset_index()

    if group_cols:
        df = df.groupby(group_cols)

    corr = (
        df.corr(method=method, numeric_only=True)
        .dropna(how="all", axis="index")
        .reset_index()
    )

    corr.columns.name = None  # remove name of columns index

    if return_type == "tidy_pairs":
        corr = (
            corr.melt(
                id_vars=group_cols + [sample_col],
                var_name=sample_col + "_2",
                value_name="correlation",
            )
            .rename(columns={sample_col: sample_col + "_1"})
            .dropna()
            .reset_index(drop=True)
        )

    elif return_type != "matrix":
        raise ValueError(f"invalid `return_type` of {return_type}")

    return corr


if __name__ == "__main__":
    import doctest

    doctest.testmod()
