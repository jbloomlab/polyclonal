"""
==========
bootstrap
==========

Defines :class:`PolyclonalCollection` objects for bootstrapping :class:`Polyclonal` model parameters.

"""

import pandas as pd


def create_bootstrap_sample(df, group_by_col='concentration'):
    """ Return a bootstrapped sample of a pandas data frame, maintaining the
    same number of items when grouped by a given column.

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
        raise KeyError(f'{group_by_col} is not in provided data frame.')

    boot_df = []

    if group_by_col is not None:
        grouped_df = df.groupby(group_by_col)

        # Sample each concentration separately
        for name, group in grouped_df:
            boot_df.append(group.sample(n=len(group), replace=True))
    else:
        boot_df.append(df.sample(n=len(df), replace=True))

    return pd.concat(boot_df)


def create_bootstrap_polyclonal():
    """
    Creates a :class:Polyclonal object from bootstrapped dataset and fits model.
    Then re-initializes that model without the attached dataframe and just wt_activit_df and mut_escape_df from the big model.
    Returns the "slim" model.
    """
    pass


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



    def fit_models(n_threads=1):
        """
        Fits :class:Polyclonal models.
        Initializes models with bootstrapped `data_to_fit`, and then fits model.

        After this fitting, re-initialize :class:Polyclonal without data, but
        with inferred parameters.

        Save the model without attached data to `self.models`.

        @Zorian given that we'd like to use multiprocessing.Pool I can say from
        experience that your life will be much easier if you think about defining things
        as much as possible in terms of free functions. I suggest a function that takes
        a bootstrapped data set and spits out a re-initialized model like you describe here.
        Then fit_models can take a threads argument, initialize the pool, and then apply
        that free function repeatedly.
        Given this, I'm not sure that you need _populate_collection but I probably don't
        understand what you have in mind.

        @Erick okay, I do think this is a good idea and have adjusted things above.
        I guess one thing I'm confused by "free function" -- do you mean a non-member function here
        or just like, more modularized functions that would all get called here?
        Regardless, I have a seperate function that will:
         - Take in a dataframe for `data_to_fit` and then creates and fits a polyclonal model.
         - Re-creates this object without the attached data and just the params.
         - Returns this polyclonal object
        We don't need _populate_collection and I've removed it.

        """
        pass

    def make_predictions():
        """
        Make predictions on variants for models that have parameters for present
        mutations.
        Aggregate and return these predictions into a single data frame.
        @Zorian-- can you provide a little more detail about the shape of this df?

        @Erick After some thought, I think I'd like a method that works like fit_models() perhaps
            * We could give a number of threads for making these predictions across all models.
            * For the target-data, we will generate the standard output for `polyclonal.prob_escape()`
            * One change here, is for variants with unseen mutations, I'd like a null prediction
            * Then we aggregate predictions:
                - We start by concatenating these `polyclonal.variant_df` objects (the model predictions)
                - We can then aggregate these predictions into a summary df for plotting using _aggregate_predictions()
        """
        pass


    def _aggregate_predictions():
        """
        Aggregate predictions from all eligible models.
        @Zorian-- can you describe the return type here?

        @Erick My plan was for the return to be a dataframe with shape N_test_variants(including each concentration) x N_summary_stats.
            * Each row would represent a variant in the "test" set.
            * Each column would be a summary statsitic of the model predictions
                - i.e., mean, number or % of models we have a prediction for (support), variance, etc.
            * And this would be the final output from `make_predictions()`
            * Though I would probably want this to look more like an `mut_escape_df` object for plotting downstream.

        """
        pass

    def mut_escape_bootstrap_heatmaps():
        """
        Visualize uncertainty in beta coefficients.
        Jesse suggests either circle size (which may get kind of hard to digest)
        or reporting in the heatmap.
        I am leaning towards the latter -- reporting the variance in the
        sampling distribution for each beta in each epitope.
        @Zorian-- How do you propose reporting?

        @Erick-- The current altair heatmaps created by polyclonal objects are really nice.
        I Think we should keep this format, but creating a "bootstrapped" heatmap where we also give the variance of each beta as well.
        So when hovering over a mutation, the user would see the mean beta-values for each epitope (same as polyclonal now), as well as the variances for these means.
        For each mutation, include the number (or frequency) of times it was
        **not** sampled as well.
        Hopefully this isn't too messy.

        A more visual altΩernative could be to use a heatmap for mean beta values
        and a seperate, epitope-wise set of heatmaps where the color corresponds
        to uncertainty/variance for that beta (the darker the color, the more uncertainty)
        """
        pass

    def activity_wt_violinplot():
        """
        Visualize the distributions of inferred WT activities across models.

        This may be tricky if we don't start the model in a good spot, as
        epitope identifiability becomes an issue.
        """
        pass
