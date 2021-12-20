"""
==========
bootstrap
==========

Defines :class:`PolyclonalCollection` objects for bootstrapping :class:`Polyclonal` model paramers.

"""

import pandas as pd


def create_bootstrap_sample(df, group_by_col='concentration'):
    """ Return a bootstrapped sample of a pandas data frame.

    Args:
        - df --> a pandas data frame to be bootstrapped
        - group_by_col (string) : the name of a column to group the dataframe by

    Returns:
        - bootstrap_df (pd.dataframe) : a dataframe that has the same number of rows as df as well as the same number of samples per group_by_col

    """
    # Check to make sure group_by_col exists -- raise an error otherwise.
    if group_by_col not in df.columns:
        raise KeyError(f'{group_by_col} is not in provdied data frame.')

    boot_df = []
    grouped_df = df.groupby(group_by_col)

    # Sample each concentration seperately
    for name, group in grouped_df:
        boot_df.append(group.sample(n=len(group), replace=True))

    # Return the actual dataframe now
    boot_df = pd.concat(boot_df)

    return boot_df
