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


class PolyclonalCollection:
    r""" A container class for multiple :class:`Polyclonal` objects.

    Parameters
    -----------
    data_to_fit : pandas.DataFrame or None
        The full dataset to draw bootstrapped samples from.
        Should have columns named 'aa_substitutions', 'concentration', and
        'prob_escape'. The 'aa_substitutions' column defines each variant
        :math:`v` as a string of substitutions (e.g., 'M3A K5G'). The
        'prob_escape' column gives the :math:`p_v\left(c\right)` value for
        each variant at each concentration :math:`c`.
    n_samples : int
        The number of desired bootstrap samples to draw and :class:Polyclonal
        models to fit.
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


    Attributes
    -----------
    mutations : tuple
        All mutations for which we have escape values.
    data_to_fit : pandas.DataFrame or None
        Data to fit as passed when initializing this :class:`BinaryMap`.
    models : tuple
        Contains `n_samples` of the :class:`Polyclonal` models.
    unsampled_mutations : dictionary
        A dictionary that keeps track of which mutations are not seen by a model
        and how often they are not sampled across all models in the collection.

    """

    def __init__(self,
                 *,
                 data_to_fit=None,
                 n_samples=None,
                 activity_wt_df=None,
                 mut_escape_df=None,
                 site_escape_df=None,
                 n_epitopes=None,
                ):
        """ See main class docstring and :class:Polyclonal documentation for
        details."""
        pass

    @staticmethod
    def _populate_collection():
        """
        Creates `n_samples` :class:Polyclonal objects, each with a different
        bootstrapped dataset.
        """
        pass

    def fit_models():
        """
        Fits :class:Polyclonal models.
        Initializes models with bootstrapped `data_to_fit`, and then fits model.

        After this fitting, re-initialize :class:Polyclonal without data, but
        with inferred parameters.

        Save the model without attached data to `self.models`.
        """
        pass

    def make_predictions():
        """
        Make predictions on variants for models that have parameters for present
        mutations.
        Aggregate and return these predictions into a single data frame.
        """
        pass

    def _aggregate_predictions():
        """
        Aggregate predictions from all eligble models.
        """
        pass

    def mut_escape_bootstrap_heatmaps():
        """
        Visualize uncertainty in beta coefficients.
        Jesse suggests either circle size (which may get kind of hard to digest)
        or reporting in the heatmap.
        I am leaning towards the latter -- reporting the variance in the
        sampling distribution for each beta in each epitope.

        For each `beta`, include the number (or frequency) of times it was
        **not** sampled.
        """
        pass

    def activity_wt_violinplot():
        """
        Visualize the distributions of inferred WT activities across models.

        This may be tricky if we don't start the model in a good spot, as
        epitope identifiability becomes an issue.
        """
        pass
