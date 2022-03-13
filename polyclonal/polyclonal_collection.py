"""
======================
polyclonal_collection
======================

Defines :class:`PolyclonalCollection` objects for bootstrapping
:mod:`~polyclonal.polyclonal.Polyclonal` model parameters.

"""

from collections import Counter
from functools import partial
from itertools import repeat
import multiprocessing

import pandas as pd

import polyclonal
from polyclonal.polyclonal import PolyclonalFitError, PolyclonalHarmonizeError


class PolyclonalCollectionFitError(Exception):
    """Error fitting in :meth:`PolyclonalCollection.fit_models`."""

    pass


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

    Returns
    -------
    bootstrap_df : pandas.DataFrame
         A dataframe that has the same number of rows as `df` as well as the same
         number of samples per `group_by_col`.

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
    """Create :class:`~polyclonal.polyclonal.Polyclonal` object from bootstrapped
    dataset and fits model.

    Parameters
    -----------
    root_polyclonal : :class:`~polyclonal.polyclonal.Polyclonal`
        Initialized :class:`~polyclonal.polyclonal.Polyclonal` object with full dataset.
    seed : int
        Random seed
    groups: string
        The column name to group `root_polyclonal.data_to_fit` by,
        in most cases, this will be `concentration`

    Returns
    -------
    polyclonal : class:`~polyclonal.polyclonal.Polyclonal`
        New object from bootstrapped sample of `root_polyclonal.data_to_fit`.

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
    """Fit the model in a :class:`~polyclonal.polyclonal.Polyclonal` object.

    A wrapper method for fitting models with `multiprocessing`. If optimization
    optimization fails, :class:`~polyclonal.polyclonal.Polyclonal` objects will throw a
    :class:`~polyclonal.polyclonal.PolyclonalFitError`.

    We catch this error and proceed with the program by returning `None` for the
    model that failed optimization.

    Parameters
    ----------
    polyclonal_obj : class:`~polyclonal.polyclonal.Polyclonal`
        An initialized :class:`~polyclonal.polyclonal.Polyclonal` object.
    **kwargs
        Keyword arguments for :meth:`polyclonal.polyclonal.Polyclonal.fit`

    Returns
    -------
    polyclonal_obj : class:`~polyclonal.polyclonal.Polyclonal`
        `polyclonal_obj` but with optimized model parameters after fitting.

    """
    try:
        _ = polyclonal_obj.fit(**kwargs)
    except PolyclonalFitError:
        return None

    return polyclonal_obj


def _prob_escape_static(polyclonal_obj, variants_df):
    """Make escape probability predictions for a dataframe of variants, given a
    polyclonal object.

    Parameters
    ----------
    polyclonal_obj : :class:`~polyclonal.polyclonal.Polyclonal`
        A :class:`~polyclonal.polyclonal.Polyclonal` object to make predictions with.

    vairants_df : pandas.DataFrame
        A dataframe of variants to predict escape probabilities for.

    Returns
    -------
    variants_df : pandas.DataFrame
        A dataframe of the variants from the input data and corresponding
        predictions for escape probabilities.

    """
    return polyclonal_obj.prob_escape(variants_df=variants_df)


class PolyclonalCollection:
    r"""A container class for multiple :class:`~polyclonal.polyclonal.Polyclonal` objects.

    Parameters
    -----------
    root_polyclonal : :class:`~polyclonal.polyclonal.Polyclonal`
        The polyclonal object created with the full dataset to draw bootstrapped
        samples from.
    n_bootstrap_samples : int
        Number of bootstrapped :class:`~polyclonal.polyclonal.Polyclonal` models to fit.
    seed : int
        Random seed for reproducibility.
    n_threads : int
        Number of threads to use for multiprocessing, -1 means all available.

    Attributes
    -----------
    mutations : tuple
        All mutations for which we have escape values.
    models : tuple
        `n_bootstrap_samples` of :class:`~polyclonal.polyclonal.Polyclonal` models.
    unsampled_mutations : dictionary
        A dictionary that keeps track of which mutations that are not seen by
        at least one model. The keys are the mutations and the values are the
        number of models that did not observe the mutation.
    n_threads: int
        Number of threads for multiprocessing.

    """

    def __init__(
        self,
        root_polyclonal,
        n_bootstrap_samples,
        n_threads=-1,
        seed=0,
    ):
        """See main class docstring for details."""
        if root_polyclonal.data_to_fit is None:
            raise ValueError("polyclonal object does not have data to fit.")
        self.root_polyclonal = root_polyclonal
        self.n_bootstrap_samples = n_bootstrap_samples
        if n_threads == -1:
            self.n_threads = multiprocessing.cpu_count()
        else:
            self.n_threads = n_threads
        self.next_seed = seed + self.n_bootstrap_samples  # For retrying

        if self.n_bootstrap_samples > 0:
            # Create distinct seeds for each model
            seeds = range(seed, self.n_bootstrap_samples)

            # Create list of bootstrapped polyclonal objects
            with multiprocessing.Pool(self.n_threads) as p:
                self.models = p.starmap(
                    _create_bootstrap_polyclonal,
                    zip(repeat(root_polyclonal), seeds),
                )
        else:
            raise ValueError("Please specify a number of bootstrap samples to make.")

    def fit_models(self, failures="error", **kwargs):
        """Fits :class:`~polyclonal.polyclonal.Polyclonal` models.
        Initializes models with bootstrapped `data_to_fit`, and then fits model.

        After fitting, re-initialize :class:`~polyclonal.polyclonal.Polyclonal` without
        data, but with inferred parameters.

        Save the models without attached data to :attr:`PolyclonalCollection.models`.

        Parameters
        ----------
        failures : {"error", "tolerate"}
            Tolerate failures in model fitting or raise an error if a failure?
            Always raise an error if all models failed.
        **kwargs
            Keyword arguments for :meth:`polyclonal.polyclonal.Polyclonal.fit`

        Returns
        -------
        (n_fit, n_failed)
            Number of model fits that failed and succeeded.

        """
        # Initial pass over all models
        with multiprocessing.Pool(self.n_threads) as p:
            self.models = p.map(
                partial(_fit_polyclonal_model_static, **kwargs), self.models
            )

        # Check to see how many models failed optimization
        n_failed = sum(model is None for model in self.models)
        if failures == "error":
            if n_failed:
                raise PolyclonalCollectionFitError(
                    f"Failed fitting {n_failed} of {len(self.models)} models"
                )
        elif failures != "tolerate":
            raise ValueError(f"invalid {failures=}")
        n_fit = len(self.models) - n_failed
        if n_fit == 0:
            raise PolyclonalCollectionFitError(
                f"Failed fitting all {len(self.models)} models"
            )

        self.models = [m for m in self.models if m is not None]
        for m in self.models:
            m.harmonize_epitopes_with(self.root_polyclonal)

        return (n_fit, n_failed)

    def make_predictions(self, variants_df):
        """Make predictions on variants for models that have parameters for
        present mutations.
        Aggregate and return these predictions into a single data frame.

        Parameters
        ----------
        variants_df : pandas.DataFrame
            Data frame defining variants. Should have column named
            ‘aa_substitutions’ that defines variants as space-delimited strings
            of substitutions (e.g., ‘M1A K3T’).

        Returns
        -------
        pred_dfs : list of pandas.DataFrame objects
            For each model in the `PolyclonalCollection`, generates a dataframe
            of predictions on `variants_df` and returns them in a list.

        """
        with multiprocessing.Pool(self.n_threads) as p:
            pred_dfs = p.starmap(
                _prob_escape_static, zip(self.models, repeat(variants_df))
            )
        return pred_dfs

    def summarize_bootstraped_predictions(self, pred_list):
        """Aggregate predictions from all eligible models.
        Given a list of prediction dataframes, splits each variant up by each
        mutation, and calculates summary statistics of escape predictions
        associated with each mutation at each concentration.

        Parameters
        ----------
        pred_list : list of pandas.DataFrame objects

        Returns
        -------
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

        Returns
        -------
        mut_escape_stats_dict : dict
            A dictionary of dataframes in the format of `self.mut_escape_df`,
            but instead have a summary statistic for the beta parameter for the
            coresponding mutation. Statistic is given by the key in the object.
        activity_wt_stats_dict : dict
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
