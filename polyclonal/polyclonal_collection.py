"""
======================
polyclonal_collection
======================

Defines :class:`PolyclonalCollection` for handling collections of multiple
:mod:`~polyclonal.polyclonal.Polyclonal` objects.

:class:`PolyclonalCollection` is a base class for the following specific use-case
classes:

 - :class:`PolyclonalBootstrap` for bootstrapping a model.

"""

import copy
import multiprocessing
from functools import partial
from itertools import repeat

import pandas as pd

import polyclonal
import polyclonal.plot
from polyclonal.polyclonal import PolyclonalFitError


class PolyclonalCollectionFitError(Exception):
    """Error fitting models."""

    pass


def create_bootstrap_sample(
    df,
    seed=0,
    group_by_col="concentration",
    sample_by="barcode",
):
    """Bootstrap sample of data frame.

    Parameters
    -----------
    df : pandas.DataFrame
        Dataframe to be bootstrapped
    seed : int
        Random number seed.
    group_by_col : string or None
        Group by this column and bootstrap each group separately.
    sample_by : str or None
        For each group, sample the same entities in this column. Requires
        each group to have same unique set of rows for this column.

    Returns
    -------
    bootstrap_df : pandas.DataFrame
         Dataframe with same number of rows as `df` and same number of samples
         per `group_by_col`.

    Example
    -------
    >>> df_groups_same_barcode = pd.DataFrame({
    ...     "aa_substitutions": ["", "M1A", "G2C", "", "M1A", "G2C"],
    ...     "concentration": [1, 1, 1, 2, 2, 2],
    ...     "barcode": ["AA", "AC", "AG", "AA", "AC", "AG"],
    ... })

    Same variants for each concentration:

    >>> create_bootstrap_sample(df_groups_same_barcode)
      aa_substitutions  concentration barcode
    0                               1      AA
    1              M1A              1      AC
    2                               1      AA
    3                               2      AA
    4              M1A              2      AC
    5                               2      AA

    Different variants for each concentration:

    >>> create_bootstrap_sample(df_groups_same_barcode, sample_by=None, seed=2)
      aa_substitutions  concentration barcode
    0                               1      AA
    1              M1A              1      AC
    2                               1      AA
    3              G2C              2      AG
    4                               2      AA
    5              M1A              2      AC

    Can't use `sample_by` if concentrations don't have same barcodes:

    >>> create_bootstrap_sample(df_groups_same_barcode.head(5))
    Traceback (most recent call last):
     ...
    ValueError: elements in sample_by='barcode' differ in group_by_col='concentration'

    """
    # if no group_by_col, make dummy one so we can use same code for both cases
    dummy_group_by_col = "_dummy_group_by"
    if dummy_group_by_col in df.columns:
        raise ValueError(f"{df.columns=} cannot have column {dummy_group_by_col=}")
    if group_by_col is None:
        group_by_col = dummy_group_by_col
    elif group_by_col not in df.columns:
        raise ValueError(f"{group_by_col=} not in {df.columns=}")

    if sample_by is not None:
        # check sample_by same for all groups and unique within groups
        samples_by_group = df.groupby(group_by_col).aggregate(
            n_rows=pd.NamedAgg(sample_by, "count"),
            n_unique=pd.NamedAgg(sample_by, "nunique"),
            samples=pd.NamedAgg(sample_by, set),
        )
        if not (samples_by_group["n_rows"] == samples_by_group["n_unique"]).all():
            raise ValueError(f"elements in {sample_by=} not unique in {group_by_col=}")
        samples = samples_by_group["samples"].values[0]
        if any(samples != s for s in samples_by_group["samples"]):
            raise ValueError(f"elements in {sample_by=} differ in {group_by_col=}")

    boot_df = []
    for i, (_, group) in enumerate(df.groupby(group_by_col)):
        if sample_by and boot_df:
            # get same sample_by as in first data frame
            boot_df.append(
                boot_df[0][[sample_by]].merge(
                    group,
                    how="left",
                    on=sample_by,
                    validate="many_to_one",
                )
            )
        else:
            boot_df.append(
                group.sample(n=len(group), replace=True, random_state=seed + i)
            )

    return pd.concat(boot_df, ignore_index=True).drop(
        columns=dummy_group_by_col, errors="ignore"
    )


def _create_bootstrap_polyclonal(
    root_polyclonal,
    seed=0,
    sample_by="barcode",
    group_by_col="concentration",
):
    """Create :class:`~polyclonal.polyclonal.Polyclonal` object from bootstrapped
    dataset and fits model. The model is initialized to the parameters in
    `root_polyclonal`.

    Parameters
    -----------
    root_polyclonal : :class:`~polyclonal.polyclonal.Polyclonal`
        Initialized :class:`~polyclonal.polyclonal.Polyclonal` object with full dataset.
    seed : int
        Random seed
    sample_by : str
        Passed to :func:`create_bootstrap_sample`.
    group_by_col: str
        Passed to :func:`create_bootstrap_sample`.

    Returns
    -------
    :class:`~polyclonal.polyclonal.Polyclonal`
        New object from bootstrapped sample of `root_polyclonal.data_to_fit`.

    """
    if root_polyclonal.data_to_fit is None:
        raise ValueError("No data to fit provided in the polyclonal object.")

    bootstrap_df = create_bootstrap_sample(
        df=root_polyclonal.data_to_fit,
        seed=seed,
        sample_by=sample_by,
        group_by_col=group_by_col,
    )

    return polyclonal.Polyclonal(
        activity_wt_df=root_polyclonal.activity_wt_df,
        mut_escape_df=root_polyclonal.mut_escape_df,
        data_to_fit=bootstrap_df,
        collapse_identical_variants=False,
        alphabet=root_polyclonal.alphabet,
        epitope_colors=root_polyclonal.epitope_colors,
        data_mut_escape_overlap="prune_to_data",  # some muts maybe not in bootstrap
    )


def _fit_func(model, **kwargs):
    """Fit model as utility function for `fit_models`."""
    try:
        _ = model.fit(**kwargs)
        return model
    except PolyclonalFitError:
        return None


def fit_models(models, n_threads, failures="error", **kwargs):
    """Fit collection of :class:`~polyclonal.polyclonal.Polyclonal` models.

    Enables fitting of multiple models simultaneously using multiple threads.

    Parameters
    ----------
    models : list
        List of :class:`~polyclonal.polyclonal.Polyclonal` models to fit.
    n_threads : int
        Number of threads (CPUs, cores) to use for fitting. Set to -1 to use
        all CPUs available.
    failures : {"error", "tolerate"}
        What if fitting fails for a model? If "error" then raise an error,
        if "ignore" then just return `None` for models that failed optimization.
    **kwargs
        Keyword arguments for :meth:`polyclonal.polyclonal.Polyclonal.fit`.

    Returns
    -------
    (n_fit, n_failed, fit_models)
        Number of models that fit successfully, number of models that failed,
        and list of the fit models. Since :class:`~polyclonal.polyclonal.Polyclonal` are
        mutable, you can also access the fit models in their original data structure.

    """
    if n_threads == -1:
        n_threads = multiprocessing.cpu_count()

    with multiprocessing.Pool(n_threads) as p:
        fit_models = p.map(partial(_fit_func, **kwargs), models)

    assert len(fit_models) == len(models)

    # Check to see if any models failed optimization
    n_failed = sum(model is None for model in fit_models)
    if failures == "error":
        if n_failed:
            raise PolyclonalCollectionFitError(
                f"Failed fitting {n_failed} of {len(models)} models"
            )
    elif failures != "tolerate":
        raise ValueError(f"invalid {failures=}")
    n_fit = len(fit_models) - n_failed
    if n_fit == 0:
        raise PolyclonalCollectionFitError(f"Failed fitting all {len(models)} models")

    return n_fit, n_failed, fit_models


class PolyclonalCollection:
    r"""Handle a collection of :class:`~polyclonal.polyclonal.Polyclonal` objects.

    Parameters
    -----------
    models_df : pandas.DataFrame
        Data frame of models. Should have one column named "model" that has
        :class:`~polyclonal.polyclonal.Polyclonal` models, and other columns
        are descriptor for model (e.g., "replicate", etc). The descriptors
        for each row must be unique.

    Attributes
    -----------
    models : list
        List of the models in `models_df`.
    model_descriptors : dict
        A list of same length as `models` with each entry being a dict keyed
        by descriptors and values being the descriptor for that model. All models
        must have same descriptor labels. Eg, ``[{"replicate": 1}, {"replicate": 2}]```.
        The descriptor labels are all columns in `models_df` except one named "model".
    descriptor_names : list
        The names that key the entries in :attr:`PolyclonalCollection.model_descriptors`.
    epitope_colors : dict
        Same meaning as for :attr:`~polyclonal.polyclonal.Polyclonal.epitope_colors`,
        extracted from :attr:`PolyclonalCollection.models`.
    alphabet : array-like
        Same meaning as for :attr:`~polyclonal.polyclonal.Polyclonal.alphabet`,
        extracted from :attr:`PolyclonalCollection.models`.

    """

    def __init__(self, models_df):
        """See main class docstring for details."""
        self.models = models_df["model"].tolist()
        if not (
            len(self.models) > 0 and len([m for m in self.models if m is not None])
        ):
            raise ValueError(f"No non-None models:\n{models_df=}")

        descriptors_df = models_df.drop(columns="model").reset_index(drop=True)
        if not len(descriptors_df.columns):
            raise ValueError("not descriptor columns in `models_df`")
        self.descriptor_names = models_df.columns.tolist()
        if len(descriptors_df.drop_duplicates()) != len(self.models):
            raise ValueError("some models have the same descriptors")
        self.model_descriptors = list(descriptors_df.to_dict(orient="index").values())

        for attr in ["epitope_colors", "alphabet"]:
            for model in self.models:
                if model is not None:
                    if not hasattr(self, attr):
                        setattr(self, attr, copy.copy(getattr(model, attr)))
                    elif getattr(self, attr) != getattr(model, attr):
                        raise ValueError(f"{attr} not the same for all models")

    @property
    def activity_wt_df_replicates(self):
        """pandas.DataFrame: Epitope activities for all models."""
        return pd.concat(
            [
                m.activity_wt_df.assign(**desc)
                for m, desc in zip(self.models, self.model_descriptors)
                if m is not None
            ],
            ignore_index=True,
        )

    @property
    def activity_wt_df(self):
        """pandas.DataFrame: Epitope activities summarized across models."""
        return self.activity_wt_df_replicates.groupby(
            "epitope", as_index=False
        ).aggregate(
            activity_mean=pd.NamedAgg("activity", "mean"),
            activity_std=pd.NamedAgg("activity", "std"),
        )

    def activity_wt_barplot(self, **kwargs):
        """Bar plot of epitope activities mean across models.

        Parameters
        ----------
        **kwargs
            Keyword arguments for :func:`polyclonal.plot.activity_wt_barplot`.

        Returns
        -------
        altair.Chart
            Interactive plot, with error bars showing standard deviation.

        """
        return polyclonal.plot.activity_wt_barplot(
            activity_wt_df=self.activity_wt_df,
            epitope_colors=self.epitope_colors,
            stat="activity_mean",
            error_stat="activity_std",
            **kwargs,
        )

    @property
    def mut_escape_df_replicates(self):
        """pandas.DataFrame: Mutation escape by model."""
        return pd.concat(
            [
                m.mut_escape_df.assign(**desc)
                for m, desc in zip(self.models, self.descriptors)
                if m is not None
            ],
            ignore_index=True,
        )

    @property
    def mut_escape_df(self):
        """pandas.DataFrame: Mutation escape summarized across models."""
        n_fit = sum(m is not None for m in self.models)
        return (
            self.mut_escape_df_replicates.groupby(
                ["epitope", "site", "wildtype", "mutant", "mutation"],
                as_index=False,
            )
            .aggregate(
                escape_mean=pd.NamedAgg("escape", "mean"),
                escape_std=pd.NamedAgg("escape", "std"),
                n_models=pd.NamedAgg("escape", "count"),
                times_seen=pd.NamedAgg("times_seen", "mean"),
            )
            .assign(
                frac_models=lambda x: x["n_models"] / n_fit,
            )
        )

    def mut_escape_heatmap(self, **kwargs):
        """Heatmaps of mutation escape values.

        Parameters
        ----------
        **kwargs
            Keyword args for :func:`polyclonal.plot.mut_escape_heatmap`

        Returns
        -------
        altair.Chart
            Interactive heat maps.

        """
        if "addtl_tooltip_stats" not in kwargs:
            kwargs["addtl_tooltip_stats"] = ["times_seen"]
        return polyclonal.plot.mut_escape_heatmap(
            mut_escape_df=self.mut_escape_df,
            alphabet=self.alphabet,
            epitope_colors=self.epitope_colors,
            stat="escape_mean",
            error_stat="escape_std",
            **kwargs,
        )

    def mut_escape_site_summary_df_replicates(self, **kwargs):
        """Site-level summaries of mutation escape for models.

        Parameters
        ----------
        **kwargs
            Keyword arguments to
            :math:`~polyclonal.polyclonal.Polyclonal.mut_escape_site_summary_df`.

        Returns
        -------
        pandas.DataFrame

        """
        return pd.concat(
            [
                m.mut_escape_site_summary_df(**kwargs).assign(**desc)
                for m, desc in zip(self.models, self.model_descriptions)
                if m is not None
            ],
            ignore_index=True,
        )

    def mut_escape_site_summary_df(self, **kwargs):
        """Site-level summaries of mutation escape across models.

        Parameters
        ----------
        **kwargs
            Keyword arguments to
            :math:`~polyclonal.polyclonal.Polyclonal.mut_escape_site_summary_df`.

        Returns
        -------
        pandas.DataFrame
            The different site-summary metrics ('mean', 'total positive', etc) are
            in different rows for each site and epitope. The 'frac_models'
            column refers to models with measurements for any mutation at that site.

        """
        n_fit = sum(m is not None for m in self.models)
        return (
            self.mut_escape_site_summary_df_replicates(**kwargs)
            .melt(
                id_vars=["epitope", "site", "wildtype", *self.descriptor_names],
                var_name="metric",
                value_name="escape",
            )
            .groupby(["epitope", "site", "wildtype", "metric"], as_index=False)
            .aggregate(
                escape_mean=pd.NamedAgg("escape", "mean"),
                escape_std=pd.NamedAgg("escape", "std"),
                n_models=pd.NamedAgg("escape", "count"),
            )
            .assign(
                frac_models=lambda x: x["n_models"] / n_fit,
            )
        )

    def mut_escape_lineplot(
        self,
        *,
        mut_escape_site_summary_df_kwargs=None,
        mut_escape_lineplot_kwargs=None,
    ):
        """Line plots of mutation escape at each site.

        Parameters
        ----------
        mut_escape_site_summary_df_kwargs : dict
            Keyword args for :meth:`PolyclonalCollection.mut_escape_site_summary_df`.
            It is often useful to set `min_times_seen` to >1.
        mut_escape_lineplot_kwargs : dict
            Keyword args for :func:`polyclonal.plot.mut_escape_lineplot`

        Returns
        -------
        altair.Chart
            Interactive heat maps.

        """
        if mut_escape_site_summary_df_kwargs is None:
            mut_escape_site_summary_df_kwargs = {}
        if mut_escape_lineplot_kwargs is None:
            mut_escape_lineplot_kwargs = {}
        if "addtl_tooltip_stats" not in mut_escape_lineplot_kwargs:
            mut_escape_lineplot_kwargs["addtl_tooltip_stats"] = ["n mutations"]
        df = self.mut_escape_site_summary_df(**mut_escape_site_summary_df_kwargs)
        return polyclonal.plot.mut_escape_lineplot(
            mut_escape_site_summary_df=df,
            replicate_data=True,
            epitope_colors=self.epitope_colors,
            **mut_escape_lineplot_kwargs,
        )

    def icXX_replicates(self, variants_df, **kwargs):
        """Concentration which given fraction is neutralized (eg IC50) for all models.

        Parameters
        ----------
        variants_df : pandas.DataFrame
            Data frame defining variants. Should have column named
            'aa_substitutions' that defines variants as space-delimited
            strings of substitutions (e.g., 'M1A K3T').
        **kwargs : Dictionary
            Keyword args for :func:`~polyclonal.polyclonal.Polyclonal.icXX`

        Returns
        -------
        pandas.DataFrame
            Copy of ``variants_df`` with added column ``col`` containing icXX,
            and model descriptors. Variants with a mutation lacking in a particular
            model are missing in that row.

        """
        return pd.concat(
            [
                m.icXX(m.filter_variants_by_seen_muts(variants_df), **kwargs).assign(
                    **desc
                )
                for m, desc in zip(self.models, self.model_descriptions)
                if m is not None
            ],
            ignore_index=True,
        )

    def icXX(self, variants_df, **kwargs):
        """Predicted concentration at which a variant is neutralized across all models.

        Parameters
        ----------
        variants_df : pandas.DataFrame
            Data frame defining variants. Should have column named
            'aa_substitutions' that defines variants as space-delimited
            strings of substitutions (e.g., 'M1A K3T').
        **kwargs : Dictionary
            Keyword args for :func:`~polyclonal.polyclonal.Polyclonal.icXX`

        Returns
        -------
        pandas.DataFrame
            Copy of ``variants_df`` with added column ``col`` containing icXX,
            and summary stats for each variant across all models.

        """
        n_fit = sum(m is not None for m in self.models)
        if "col" in kwargs:
            col = kwargs["col"]
        else:
            col = "IC50"
        if len(variants_df) != len(variants_df.drop_duplicates()):
            raise ValueError("columns in `variants_df` must be unique")
        return (
            self.icXX_replicates(variants_df, **kwargs)
            .groupby(variants_df.columns.tolist(), as_index=False)
            .aggregate(
                mean_IC=pd.NamedAgg(col, "mean"),
                median_IC=pd.NamedAgg(col, "median"),
                std_IC=pd.NamedAgg(col, "std"),
                n_models=pd.NamedAgg(col, "count"),
            )
            .assign(
                frac_models=lambda x: x["n_models"] / n_fit,
            )
            .rename(
                columns={
                    f"{stat}_IC": f"{stat}_{col}" for stat in ["mean", "median", "std"]
                }
            )
        )

    def prob_escape_replicates(self, variants_df, **kwargs):
        r"""Compute predicted probability of escape :math:`p_v\left(c\right)`.

        Uses all models to make predictions on ``variants_df``.

        Arguments
        ---------
        variants_df : pandas.DataFrame
            Input data frame defining variants. Should have a column
            named 'aa_substitutions' that defines variants as space-delimited
            strings of substitutions (e.g., 'M1A K3T'). Should also have a
            column 'concentration' if ``concentrations=None``.
        **kwargs : Dictionary
            Keyword args for :func:`~polyclonal.polyclonal.Polyclonal.prob_escape`

        Returns
        -------
        pandas.DataFrame
            Version of ``variants_df`` with columns named 'concentration'
            and 'predicted_prob_escape' giving predicted probability of escape
            :math:`p_v\left(c\right)` for each variant at each concentration and
            model. Variants with a mutation lacking in a particular model are
            missing in that row.

        """
        return pd.concat(
            [
                m.prob_escape(
                    variants_df=m.filter_variants_by_seen_muts(variants_df),
                    **kwargs,
                ).assign(**desc)
                for m, desc in enumerate(self.models, self.model_descriptions)
                if m is not None
            ],
            ignore_index=True,
        )

    def prob_escape(self, variants_df, **kwargs):
        r"""Summary of predicted probability of escape across all models.

        Arguments
        ---------
        variants_df : pandas.DataFrame
            Input data frame defining variants. Should have a column
            named 'aa_substitutions' that defines variants as space-delimited
            strings of substitutions (e.g., 'M1A K3T'). Should also have a
            column 'concentration' if ``concentrations=None``.
        **kwargs : Dictionary
            Keyword args for :func:`~polyclonal.polyclonal.Polyclonal.prob_escape`

        Returns
        -------
        pandas.DataFrame
            Version of ``variants_df`` with columns named 'concentration'
            and 'mean', 'median', and 'std' giving corresponding summary stats
            of predicted probability of escape :math:`p_v\left(c\right)`
            for each variant at each concentration across models.

        """
        n_fit = sum(m is not None for m in self.models)
        if len(variants_df) != len(variants_df.drop_duplicates()):
            raise ValueError("columns in `variants_df` must be unique")
        return (
            self.prob_escape_replicates(variants_df=variants_df, **kwargs)
            .groupby(variants_df.columns.tolist(), as_index=False)
            .aggregate(
                mean_predicted_prob_escape=pd.NamedAgg("predicted_prob_escape", "mean"),
                median_predicted_prob_escape=pd.NamedAgg(
                    "predicted_prob_escape",
                    "median",
                ),
                std_predicted_prob_escape=pd.NamedAgg("predicted_prob_escape", "std"),
                n_models=pd.NamedAgg("predicted_prob_escape", "count"),
            )
            .assign(
                frac_models=lambda x: x["n_models"] / n_fit,
            )
        )


class PolyclonalBootstrap(PolyclonalCollection):
    r"""Bootstrap :class:`~polyclonal.polyclonal.Polyclonal` objects.

    Parameters
    -----------
    root_polyclonal : :class:`~polyclonal.polyclonal.Polyclonal`
        The polyclonal object created with the full dataset to draw bootstrapped
        samples from. The bootstrapped samples are also initialized to mutation effects
        and activities of this model, so it is **highly recommended** that this object
        already have been fit to the full dataset.
    n_bootstrap_samples : int
        Number of bootstrapped :class:`~polyclonal.polyclonal.Polyclonal` models to fit.
    seed : int
        Random seed for reproducibility.
    n_threads : int
        Number of threads to use for multiprocessing, -1 means all available.
    sample_by
        Passed to :func:`create_bootstrap_sample`. Should generally be 'barcode'
        if you have same variants at all concentrations, and maybe `None` otherwise.

    Attributes
    -----------
    root_polyclonal : :class:`~polyclonal.polyclonal.Polyclonal`
        The root polyclonal object passed as a parameter.
    n_threads: int
        Number of threads for multiprocessing.
    Other attributes of :class:`PolyclonalCollection`.
        Inherited from base class.

    """

    def __init__(
        self,
        root_polyclonal,
        n_bootstrap_samples,
        n_threads=-1,
        seed=0,
        sample_by="barcode",
    ):
        """See main class docstring for details."""
        if root_polyclonal.data_to_fit is None:
            raise ValueError("polyclonal object does not have data to fit.")
        self.root_polyclonal = root_polyclonal
        if n_threads == -1:
            self.n_threads = multiprocessing.cpu_count()
        else:
            self.n_threads = n_threads

        if n_bootstrap_samples > 0:
            # Create distinct seeds for each model
            seeds = range(seed, seed + n_bootstrap_samples)

            # Create list of bootstrapped polyclonal objects
            with multiprocessing.Pool(self.n_threads) as p:
                models = p.starmap(
                    _create_bootstrap_polyclonal,
                    zip(repeat(root_polyclonal), seeds, repeat(sample_by)),
                )
        else:
            raise ValueError("Please specify a number of bootstrap samples to make.")

        super().__init__(
            pd.DataFrame({"model": models}).assign(
                bootstrap_replicate=lambda x: x.index + 1
            )
        )

    def fit_models(self, failures="error", **kwargs):
        """Fits bootstrapped :class:`~polyclonal.polyclonal.Polyclonal` models.

        The fit models will then be in :attr:`PolyclonalCollection.models`,
        with any models that fail fitting set to `None`.

        Parameters
        ----------
        failures : {"error", "tolerate"}
            Tolerate failures in model fitting or raise an error if a failure?
            Always raise an error if all models failed.
        **kwargs
            Keyword arguments for :meth:`polyclonal.polyclonal.Polyclonal.fit`.
            If not specified otherwise, `fit_site_level_first` is set to `False`,
            since models are initialized to "good" values from the root object.

        Returns
        -------
        (n_fit, n_failed)
            Number of model fits that failed and succeeded.

        """
        if "fit_site_level_first" not in kwargs:
            kwargs["fit_site_level_first"] = False

        n_fit, n_failed, self.models = fit_models(
            self.models,
            self.n_threads,
            failures,
            **kwargs,
        )

        for m in self.models:
            if m is not None:
                m.harmonize_epitopes_with(self.root_polyclonal)

        return (n_fit, n_failed)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
