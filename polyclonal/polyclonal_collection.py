"""
======================
polyclonal_collection
======================

Defines :class:`PolyclonalCollection` objects for bootstrapping
:class:`Polyclonal` model parameters.

"""

from collections import Counter
from functools import partial
from itertools import repeat
from multiprocessing import Pool

import pandas as pd

import polyclonal


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
        The name of a column to group the dataframe by. In most cases, this will
        be 'concentration'

    Returns:
    ---------
    bootstrap_df : pandas.DataFrame
         A dataframe that has the same number of rows as df as well as the same
         number of samples per group_by_col

    """
    # Check to make sure group_by_col exists -- raise an error otherwise.
    if group_by_col is not None and group_by_col not in df.columns:
        raise KeyError(f"{group_by_col} is not in provided data frame.")

    boot_df = []

    if group_by_col is not None:
        grouped_df = df.groupby(group_by_col)

        # Sample each concentration separately
        for _, group in grouped_df:
            boot_df.append(group.sample(n=len(group), replace=True, random_state=seed))
    else:
        boot_df.append(df.sample(n=len(df), replace=True, random_state=seed))

    return pd.concat(boot_df)


def _create_bootstrap_polyclonal(root_polyclonal, seed=0, group_by_col="concentration"):
    """Create a :class:Polyclonal object from bootstrapped dataset and
    fits model.

    Parameters
    -----------
    root_polyclonal : :class:Polyclonal
        A initialized :class:Polyclonal object with complete dataset.
    seed : int
        Random seed
    groups: string
        The column name to group `root_polyclonal.data_to_fit` by,
        in most cases, this will be `concentration`

    Returns:
    ---------
    polyclonal : class:Polyclonal
        A new :class:Polyclonal object constructed from a bootsrapped sample of
        `root_polyclonal.data_to_fit`.

    """
    if root_polyclonal.data_to_fit is None:
        raise ValueError("No data to fit provided in the polyclonal object.")

    bootstrap_df = create_bootstrap_sample(
        df=root_polyclonal.data_to_fit, seed=seed, group_by_col=group_by_col
    )

    return polyclonal.Polyclonal(
        data_to_fit=bootstrap_df,
        n_epitopes=len(root_polyclonal.epitopes),
        collapse_identical_variants=False,
    )


def _fit_polyclonal_model_static(polyclonal_obj, **kwargs):
    """Fit the model in a :class:Polyclonal object.

    A wrapper method for fitting models with `multiprocessing`.
    If scipy optimization fails, :class:Polyclonal objects will throw a
    `RuntimeError`.

    We catch this error and proceed with the program by returning `None` for the
    model that failed optimization.

    Parameters:
    ------------
    polyclonal_obj : class:Polyclonal
        An initialized :class:Polyclonal object.

    Returns:
    ---------
    polyclonal_obj : class:Polyclonal
        The same class:Polyclonal object but with optimized model parameters
        after fitting.

    """
    try:
        _ = polyclonal_obj.fit(**kwargs)
    except RuntimeError:
        return None

    return polyclonal_obj


def _prob_escape_static(polyclonal_obj, variants_df):
    """Make escape probability predictions for a dataframe of variants, given a
    polyclonal object.

    Parameters:
    ------------
    polyclonal_obj : :class:Polyclonal
        A :class:Polyclonal object to make predictions with.

    vairants_df : pandas.DataFrame
        A dataframe of variants to predict escape probabilities for.

    Returns:
    ---------
    variants_df : pandas.DataFrame
        A dataframe of the variants from the input data and corresponding
        predictions for escape probabilities.

    """
    return polyclonal_obj.prob_escape(variants_df=variants_df)


def _harmonize_epitopes_static(other_poly, ref_poly):
    """Harmonize the epitopes of an polycolonal object with a root_polyclonal
    object.

    A wrapper method for fitting models with `multiprocessing`.
    If mapping matricies are not 1-to-1, :class:Polyclonal objects will throw a
    `ValueError`.

    If we catch this error, the program will return a value of `None` for the
    `other_poly` model that could not be harmonized with `ref_poly`.


    Parameters:
    ------------
    other_poly : :class:Polyclonal
        Another :class:Polyclonal object that will align its epitopes with
        `ref_poly`.
    ref_poly : :class:Polyclonal
        A :class:Polyclonal object to serve as the reference object.

    Returns:
    ---------
    other_poly : :class:Polyclonal
        The same :class:Polyclonal object but with aligned epitopes with
        `ref_poly`.

    """
    try:
        other_poly.harmonize_epitopes_with(ref_poly)
    except ValueError:
        return None

    return other_poly


class PolyclonalCollection:
    r"""A container class for multiple :class:`Polyclonal` objects.

    Parameters
    -----------
    root_polyclonal : :class:Polyclonal
        The polyclonal object created with the full dataset to draw bootstrapped
        samples from.
    n_bootstrap_samples : int
        The number of desired bootstrap samples to draw and :class:Polyclonal
        models to fit.
    seed : int
        Random seed for reproducibility.
    n_threads : int
        The number of threads to use for multiprocessing.

    Attributes
    -----------
    mutations : tuple
        All mutations for which we have escape values.
    models : tuple
        Contains `n_bootstrap_samples` of the :class:`Polyclonal` models.
    unsampled_mutations : dictionary
        A dictionary that keeps track of which mutations that are not seen by
        at least one model. The keys are the mutations and the values are the
        number of models that did not observe the mutation.

    """

    def __init__(
        self,
        root_polyclonal,
        n_bootstrap_samples=0,
        n_threads=1,
        seed=0,
    ):
        """See main class docstring and :class:Polyclonal documentation for
        details.
        """
        if root_polyclonal.data_to_fit is None:
            raise ValueError("polyclonal object does not have data to fit.")
        self.root_polyclonal = root_polyclonal
        self.n_bootstrap_samples = n_bootstrap_samples
        self.n_threads = n_threads
        self.seed = seed
        self.next_seed = self.seed + self.n_bootstrap_samples  # For retrying

        if self.n_bootstrap_samples > 0:
            # Create distinct seeds for each model
            seeds = [x + self.seed for x in list(range(self.n_bootstrap_samples))]

            # Create list of bootstrapped polyclonal objects
            with Pool(self.n_threads) as p:
                self.models = p.starmap(
                    _create_bootstrap_polyclonal,
                    zip(repeat(root_polyclonal), seeds),
                )
        else:
            raise ValueError("Please specify a number of bootstrap samples to make.")

    def fit_models(self, max_attempts=10, **kwargs):
        """Fits :class:Polyclonal models.
        Initializes models with bootstrapped `data_to_fit`, and then fits model.

        After this fitting, re-initialize :class:Polyclonal without data, but
        with inferred parameters.

        Save the model without attached data to `self.models`.

        Parameters:
        ------------
        max_attempts : int
            The maximum number of retries to allow if optimization fails.

        Returns:
        ---------
        None

        """
        # Initial pass over all models
        with Pool(self.n_threads) as p:
            self.models = p.map(
                partial(_fit_polyclonal_model_static, **kwargs), self.models
            )

        # Check to see how many models failed optimization
        n_fails = sum(model is None for model in self.models)
        n_retry_fails = 0

        # Models that were optimized successfully
        replacement_models = []

        # Create replacement models one by one (for now at least)
        while len(replacement_models) < n_fails:
            if n_retry_fails == max_attempts:
                raise RuntimeError("Maximum number of fitting retries reached.")

            # Create new models one by one but only add the ones that succeed
            tmp_model = self._retry_model_fit()
            tmp_model = _harmonize_epitopes_static(self.root_polyclonal, tmp_model)

            if tmp_model is not None:
                replacement_models.append(tmp_model)
            else:
                n_retry_fails += 1

        # Now, replace all None in self.models with replacement models
        self.models = list(filter(None, self.models))
        self.models = self.models + replacement_models

    def _retry_model_fit(self):
        """Retry fitting the model in the case of failure with optimization or
        epitope harmonization.

        Returns:
        --------
        new_polyclonal : :class:Polyclonal
            A new polyclonal object with a different seed.

        """
        # Create a new model with next seed
        new_polyclonal = self._create_bootstrap_polyclonal(
            self.root_polyclonal, self.next_seed
        )
        # Fit the model again
        self._fit_polyclonal_model_static(polyclonal_obj=new_polyclonal)
        # Increment last seed
        self.next_seed += 1
        # Return model
        return new_polyclonal

    def make_predictions(self, variants_df):
        """Make predictions on variants for models that have parameters for
        present mutations.
        Aggregate and return these predictions into a single data frame.

        Parameters:
        ------------
        variants_df : pandas.DataFrame
            Data frame defining variants. Should have column named
            ‘aa_substitutions’ that defines variants as space-delimited strings
            of substitutions (e.g., ‘M1A K3T’).

        Returns:
        ---------
        pred_dfs : list of pandas.DataFrame objects
            For each model in the `PolyclonalCollection`, generates a dataframe
            of predictions on `variants_df` and returns them in a list.

        """
        with Pool(self.n_threads) as p:
            pred_dfs = p.starmap(
                _prob_escape_static, zip(self.models, repeat(variants_df))
            )
        return pred_dfs

    def summarize_bootstraped_predictions(self, pred_list):
        """Aggregate predictions from all eligible models.
        Given a list of prediction dataframes, splits each variant up by each
        mutation, and calculates summary statistics of escape predictions
        associated with each mutation at each concentration.

        Parameters:
        ------------
        pred_list : list of pandas.DataFrame objets

        Returns:
        ---------
        results_df : pandas.DataFrame
            A dataframe of summary stats for predictions made from each model.

        """
        # Combine all dataframes together
        results_df = pd.concat(pred_list)

        # Define dictionary of data transformations we want to calculate.ß
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
        """Return a dictionary of the mutations and the percentage of
        bootstrapped samples they were in.
        """
        mutation_dict = Counter()

        for model in self.models:
            # Update mutation observation counts
            mutation_dict.update(model.mutations)

        mutation_dict_freqs = {
            key: mutation_dict[key] / self.n_bootstrap_samples
            for key in mutation_dict.keys()
        }

        return mutation_dict_freqs

    def summarize_bootstrapped_params(self):
        """Return a dataframe of summary statistics for `self.mut_escape_df`
        and `self.activity_wt_df`.

        Parameters:
        ------------
        None

        Returns:
        ------------
        mut_escape_stats_dict : Dictionary
            A dictionary of dataframes in the format of `self.mut_escape_df`,
            but instead have a summary statistic for the beta parameter for the
            coresponding mutation. Statistic is given by the key in the object.
        activity_wt_stats_dict : Dictionary
            A dictionary of dataframes in the format of `self.activity_wt_df`,
            but instead have a summary statistic for the beta parameter for the
            coresponding mutation. Statistic is given by the key in the object.

        """
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

        # Create a dictionary for summary stats for both dataframes
        mut_core_cols = ["mutation", "epitope", "site", "wildtype"]

        mut_escape_stats_dict = {
            "mean": mut_escape_df_stats.filter(items=mut_core_cols + ["mean"]),
            "median": mut_escape_df_stats.filter(items=mut_core_cols + ["median"]),
            "std": mut_escape_df_stats.filter(items=mut_core_cols + ["std"]),
        }

        activity_wt_stats_dict = {
            "mean": wt_df_stats.filter(items=["epitope", "mean"]),
            "median": wt_df_stats.filter(items=["epitope", "median"]),
            "std": wt_df_stats.filter(items=["epitope", "std"]),
        }

        return mut_escape_stats_dict, activity_wt_stats_dict
