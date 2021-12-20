"""
==========
bootstrap
==========

Defines :class:`PolyclonalCollection` objects for bootstrapping :class:`Polyclonal` model paramers.

"""

import pandas as pd


def create_bootstrap_sample(df, group_by_col='concentration'):
    """ Return a bootstrapped sample of a pandas data frame.

    Parameters
    -----------
    df : pandas.DataFrame
        A dataframe to be bootstrapped
    group_by_col : string or None
        The name of a column to group the dataframe by. In most cases, this will be 'concentration'

    Returns:
    ---------
        - bootstrap_df (pd.dataframe) : a dataframe that has the same number of rows as df as well as the same number of samples per group_by_col

    """
    # Check to make sure group_by_col exists -- raise an error otherwise.
    if group_by_col is not None and group_by_col not in df.columns:
        raise KeyError(f'{group_by_col} is not in provdied data frame.')

    boot_df = []

    if group_by_col is not None:
        grouped_df = df.groupby(group_by_col)

        # Sample each concentration seperately
        for name, group in grouped_df:
            boot_df.append(group.sample(n=len(group), replace=True))
    else:
        boot_df.append(df.sample(n=len(df), replace=True))

    return pd.concat(boot_df)
