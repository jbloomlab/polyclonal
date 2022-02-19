"""
==========
bootstrap
==========

Defines :class:`PolyclonalCollection` objects for bootstrapping :class:`Polyclonal` model parameters.

"""

import pandas as pd
import polyclonal
from multiprocessing import Pool
from itertools import repeat
from collections import Counter


def create_bootstrap_sample(df, seed=0, group_by_col="concentration"):
    """Return a bootstrapped sample of a pandas data frame, maintaining the
    same number of items when grouped by a given column.

    Parameters
    -----------
    df : pandas.DataFrame
        A dataframe to be bootstrapped
    seed : int
        The random seed to use for the sample
    group_by_col : string, list or None
        The name of a column to group the dataframe by. In most cases, this will be 'concentration'

    Returns:
    ---------
    bootstrap_df : pandas.DataFrame
         A dataframe that has the same number of rows as df as well as the same number of samples per group_by_col

    """
    # Check to make sure group_by_col exists -- raise an error otherwise.
    if group_by_col is not None and group_by_col not in df.columns:
        raise KeyError(f"{group_by_col} is not in provided data frame.")

    boot_df = []

    if group_by_col is not None:
        # TODO enable grouping by multiple columns
        grouped_df = df.groupby(group_by_col)

        # Sample each concentration separately
        for name, group in grouped_df:
            boot_df.append(group.sample(n=len(group), replace=True, random_state=seed))
    else:
        boot_df.append(df.sample(n=len(df), replace=True, random_state=seed))

    return pd.concat(boot_df)


def _create_bootstrap_polyclonal(root_polyclonal, seed=0, groups="concentration"):
    """Creates a :class:Polyclonal object from bootstrapped dataset and fits model.

    Parameters
    -----------
    root_polyclonal : :class:Polyclonal
        A initialized :class:Polyclonal object with complete dataset.
    seed : int
        Random seed (for now, model number), may make this an "offset" from 0.
    groups: string
        The column name to group `root_polyclonal.data_to_fit` by, In most cases, this will be `concentration`

    Returns:
    ---------
    new_polyclonal : class:Polyclonal
        A new :class:Polyclonal object constructed from a bootsrapped sample of `root_polyclonal.data_to_fit`.

    """
    # Check to see if polyclonal has data_to_fit
    # Polyclonal drops duplicate from data_to_fit by default -- should we  keep this?
    bootstrap_df = create_bootstrap_sample(
        df=root_polyclonal.data_to_fit, seed=seed, group_by_col=groups
    )
    new_polyclonal = polyclonal.Polyclonal(
        data_to_fit=bootstrap_df,
        n_epitopes=len(root_polyclonal.epitopes),
        collapse_identical_variants=False,
    )
    # Re-initialize this model without the attached dataframe and just wt_activity_df and mut_escape_df from the big model.
    # Return the "slim" model?
    # Maybe create a new method for fitting a polyclonal model and returning the slim version?
    return new_polyclonal


def _fit_polyclonal_model(polyclonal_obj):
    """Wrapper method to fit the model in a :class:Polyclonal object.

    Parameters:
    ------------
    polyclonal_obj : class:Polyclonal
        An initialized :class:Polyclonal object.

    Returns:
    ---------
    polyclonal_obj : class:Polyclonal
        The same class:Polyclonal object but with optimized model parameters after fitting.

    """
    # TODO: How should we handle situations where optimization fails?
    # Is failed optimization only an issue on the small dataset?
    _ = polyclonal_obj.fit()
    # TODO: re-initialize that model without the attached dataframe and just info from the big model.
    return polyclonal_obj


def _predict(polyclonal_obj, variants_df):
    """Takes a polyclonal object and a dataframe of variants to predict on and makes predictions for escape probabilities.

    Parameters:
    ------------
    polyclonal_obj : :class:Polyclonal
        A :class:Polyclonal object to make predictions with.

    vairants_df : pandas.DataFrame
        A dataframe of variants to predict escape probabilities for.

    Returns:
    ---------
    variants_df : pandas.DataFrame
        A dataframe of the variants from the input data and corresponding predictions for escape probabilities.
    Returns dataframe of variants with added prob_escape column.
    """
    return polyclonal_obj.prob_escape(variants_df=variants_df)


class PolyclonalCollection:
    r"""A container class for multiple :class:`Polyclonal` objects.

    Parameters
    -----------
    root_polyclonal : :class:Polyclonal
        The polyclonal object created with the full dataset to draw bootstrapped samples from.
    n_samples : int
        The number of desired bootstrap samples to draw and :class:Polyclonal
        models to fit.
    seed : int
        Random seed for reproducibility.

    Attributes
    -----------
    mutations : tuple
        All mutations for which we have escape values.
    models : tuple
        Contains `n_samples` of the :class:`Polyclonal` models.
    unsampled_mutations : dictionary
        A dictionary that keeps track of which mutations are not seen by a model
        and how often they are not sampled across all models in the collection.

    """

    def __init__(
        self,
        root_polyclonal,
        n_samples=0,
        seed=0,
    ):
        """See main class docstring and :class:Polyclonal documentation for
        details."""
        # TODO Check to see if the polyclonal object has required args `data_to_fit`
        self.root_polyclonal = root_polyclonal
        self.n_samples = n_samples

        # TODO: in either polyclonal or in PolyclonalCollection, store indicies from original dataset.
        if n_samples > 0:
            # Create list of bootstrapped polyclonal objects (don't fit yet)
            with Pool(2) as p:
                self.models = p.starmap(
                    _create_bootstrap_polyclonal,
                    zip(repeat(root_polyclonal), list(range(n_samples))),
                )
        else:
            raise ValueError(
                "Please specify a number of bootstrap samples to make by specifying n_samples."
            )

    def fit_models(self, n_threads=1):
        """Fits :class:Polyclonal models.
        Initializes models with bootstrapped `data_to_fit`, and then fits model.

        After this fitting, re-initialize :class:Polyclonal without data, but
        with inferred parameters.

        Save the model without attached data to `self.models`.

        Parameters:
        ------------

        Returns:
        ---------

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
        with Pool(n_threads) as p:
            # May not need this re-assignment since fit() updates params in place.
            self.models = p.map(_fit_polyclonal_model, self.models)

    def make_predictions(self, variants_df, n_threads=1):
        """Make predictions on variants for models that have parameters for present
        mutations.
        Aggregate and return these predictions into a single data frame.

        Parameters:
        ------------
        variants_df : pandas.DataFrame
            Data frame defining variants. Should have column named ‘aa_substitutions’
            that defines variants as space-delimited strings of substitutions (e.g., ‘M1A K3T’).
        n_threads : int
            Number of threads to use to make predictions across all models.

        Returns:
        ---------
        pred_dfs : list of pandas.DataFrame objects
            For each model in the `PolyclonalCollection`, generates a dataframe
            of predictions on `variants_df` and returns them in a list.


        @Zorian-- can you provide a little more detail about the shape of this df?

        @Erick After some thought, I think I'd like a method that works like fit_models() perhaps
            * We could give a number of threads for making these predictions across all models.
            * For the target-data, we will generate the standard output for `polyclonal.prob_escape()`
            * One change here, is for variants with unseen mutations, I'd like a null prediction
            * Then we aggregate predictions:
                - We start by concatenating these `polyclonal.variant_df` objects (the model predictions)
                - We can then aggregate these predictions into a summary df for plotting using _aggregate_predictions()
        """
        with Pool(n_threads) as p:
            pred_dfs = p.starmap(_predict, zip(self.models, repeat(variants_df)))
        return pred_dfs

    def _aggregate_predictions(self, pred_list):
        """Aggregate predictions from all eligible models.
        Given a list of prediction dataframes, splits each variant up by each mutation,
        and calculates summary statistics of escape predictions associated with each mutation at each concentration.
        @Zorian-- can you describe the return type here?

        @Erick My plan was for the return to be a dataframe with shape N_test_variants(including each concentration) x N_summary_stats.
            * Each row would represent a variant in the "test" set.
            * Each column would be a summary statsitic of the model predictions
                - i.e., mean, number or % of models we have a prediction for (support), variance, etc.
            * And this would be the final output from `make_predictions()`
            * Though I would probably want this to look more like an `mut_escape_df` object for plotting downstream.

        Parameters:
        ------------
        pred_list : list of pandas.DataFrame objets

        Returns:
        ---------
        results_df : pandas.DataFrame
            A dataframe of summary stats for predictions made from each model.

        """
        # Combine all dataframes together (maybe add some model ID column?)
        raw_concat_df = pd.concat(pred_list)

        results_df = raw_concat_df

        pred_summary_stats = {
            "mean_predicted_prob_escape": pd.NamedAgg("predicted_prob_escape", "mean"),
            "median_predicted_prob_escape": pd.NamedAgg(
                "predicted_prob_escape", "median"
            ),
            "std_predicted_prob_escape": pd.NamedAgg("predicted_prob_escape", "std"),
            "n_model_predictions": pd.NamedAgg("prob_escape", "count"),
        }

        return results_df.groupby(
            ["barcode", "aa_substitutions", "concentration"], as_index=False, sort=False
        ).aggregate(**pred_summary_stats)

    @property
    def mut_bootstrap_freq_dict(self):
        """Gives a dictionary of the mutations and the percentage of bootstrapped samples they were in."""
        mutation_dict = Counter()

        for model in self.models:
            # Update mutation observation counts
            mutation_dict.update(model.mutations)

        mutation_dict_freqs = {
            key: mutation_dict[key] / self.n_samples for key in mutation_dict.keys()
        }

        return mutation_dict_freqs

    def _summarize_param_dfs(self):
        """Creates a dataframe of summary statistics for `self.mut_escape_df` and `self.activity_wt_df`"""

        mut_escape_df_list = []
        activity_wt_df_list = []

        # Create dictionary of desired summary stats
        escape_summary_stats = {
            "site": pd.NamedAgg("site", "first"),
            "wildtype": pd.NamedAgg("mutant", "first"),
            "mean": pd.NamedAgg("escape", "mean"),
            "median": pd.NamedAgg("escape", "median"),
            "std": pd.NamedAgg("escape", "std"),
        }

        activity_summary_stats = {
            "epitope": pd.NamedAgg("epitope", "first"),
            "mean": pd.NamedAgg("activity", "mean"),
            "median": pd.NamedAgg("activity", "median"),
            "std": pd.NamedAgg("activity", "std"),
        }

        # Grab all dataframes
        for model in self.models:
            # Add inferred params
            mut_escape_df_list.append(model.mut_escape_df)
            activity_wt_df_list.append(model.activity_wt_df)

        mut_escape_df = pd.concat(mut_escape_df_list)
        activity_wt_df = pd.concat(activity_wt_df_list)

        mut_escape_df_stats = mut_escape_df.groupby(
            ["mutation", "epitope"], as_index=False, sort=False
        ).aggregate(**escape_summary_stats)
        wt_df_stats = activity_wt_df.groupby("epitope", as_index=False).aggregate(
            **activity_summary_stats
        )

        return mut_escape_df_stats, wt_df_stats

    def avg_model(self, avg_type="median", min_bootstrap_frac=0.9):
        """Returns :class:`Polyclonal` object with average fit parameters.

        Parameters
        ------------
        avg_type : {'median', 'mean'}
            Average activities and beta values in this way.
        min_bootstrap_frac : float
            Only include beta values for mutations in this fraction of bootstrap replicates.

        Returns
        --------
        :class:`Polyclonal`
        """
        # Harmonize epitopes -- a bug here (need to raise issue)

        # Get summary stats dataframes
        avg_mut_df, avg_activity_df = self._summarize_param_dfs()

        # Filter beta values for mutations passing min_bootstrap_frac threshold
        muts_to_drop = [
            key
            for key, value in self.mut_bootstrap_freq_dict.items()
            if value < min_bootstrap_frac
        ]
        avg_mut_df = avg_mut_df[~avg_mut_df["mutation"].isin(muts_to_drop)]

        # Use desired avg_type
        avg_mut_df["escape"] = avg_mut_df[avg_type]
        avg_activity_df["activity"] = avg_activity_df[avg_type]

        # Drop all summary stat columns
        cols_to_drop = ["mean", "median", "std"]

        # Create a new polyclonal object with `mut_escape_df` of average values.
        avg_model = polyclonal.Polyclonal(
            data_to_fit=None,
            mut_escape_df=avg_mut_df.drop(columns=cols_to_drop),
            activity_wt_df=avg_activity_df.drop(columns=cols_to_drop),
        )

        # Also create columns with SD and % of bootstrap samples the mutation is in
        return avg_model
