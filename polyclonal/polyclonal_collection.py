"""
======================
polyclonal_collection
======================

Defines :class:`PolyclonalCollection` objects for bootstrapping
:mod:`~polyclonal.polyclonal.Polyclonal` model parameters.

"""

import multiprocessing
from functools import partial
from itertools import repeat

import pandas as pd

import polyclonal
import polyclonal.plot
from polyclonal.polyclonal import PolyclonalFitError


class PolyclonalCollectionFitError(Exception):
    """Error fitting in :meth:`PolyclonalCollection.fit_models`."""

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


class PolyclonalCollection:
    r"""A container class for multiple :class:`~polyclonal.polyclonal.Polyclonal` objects.

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
    models : list
        The bootstrapped :class:`~polyclonal.polyclonal.Polyclonal` models,
        will not be fit until you call :meth:`PolyclonalCollection.fit_models`.
        After fitting, if any models fail fitting they are set to `None`.
    n_threads: int
        Number of threads for multiprocessing.

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
                self.models = p.starmap(
                    _create_bootstrap_polyclonal,
                    zip(repeat(root_polyclonal), seeds, repeat(sample_by)),
                )
        else:
            raise ValueError("Please specify a number of bootstrap samples to make.")

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
        # Initial pass over all models
        if "fit_site_level_first" not in kwargs:
            kwargs["fit_site_level_first"] = False
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

        for m in self.models:
            if m is not None:
                m.harmonize_epitopes_with(self.root_polyclonal)

        return (n_fit, n_failed)

    @property
    def activity_wt_df_replicates(self):
        """pandas.DataFrame: Epitope activities for replicates"""
        return pd.concat(
            [
                m.activity_wt_df.assign(bootstrap_replicate=i)
                for i, m in enumerate(self.models, start=1)
                if m is not None
            ],
            ignore_index=True,
        )

    @property
    def activity_wt_df(self):
        """pandas.DataFrame: Epitope activities summarized across replicates."""
        return self.activity_wt_df_replicates.groupby(
            "epitope", as_index=False
        ).aggregate(
            activity_mean=pd.NamedAgg("activity", "mean"),
            activity_std=pd.NamedAgg("activity", "std"),
        )

    def activity_wt_barplot(self, **kwargs):
        """Bar plot of epitope activities mean across replicates.

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
            epitope_colors=self.root_polyclonal.epitope_colors,
            stat="activity_mean",
            error_stat="activity_std",
            **kwargs,
        )

    @property
    def mut_escape_df_replicates(self):
        """pandas.DataFrame: Mutation escape by replicate."""
        return pd.concat(
            [
                m.mut_escape_df.assign(bootstrap_replicate=i).drop(columns="times_seen")
                for i, m in enumerate(self.models, start=1)
                if m is not None
            ],
            ignore_index=True,
        )

    @property
    def mut_escape_df(self):
        """pandas.DataFrame: Mutation escape summarized across replicates."""
        n_fit = sum(m is not None for m in self.models)
        return (
            self.mut_escape_df_replicates.groupby(
                ["epitope", "site", "wildtype", "mutant", "mutation"],
                as_index=False,
            )
            .aggregate(
                escape_mean=pd.NamedAgg("escape", "mean"),
                escape_std=pd.NamedAgg("escape", "std"),
                n_bootstrap_replicates=pd.NamedAgg("bootstrap_replicate", "count"),
            )
            .assign(
                frac_bootstrap_replicates=lambda x: x["n_bootstrap_replicates"] / n_fit,
                times_seen=lambda x: x["mutation"].map(
                    self.root_polyclonal.mutations_times_seen
                ),
            )
            .drop(columns="n_bootstrap_replicates")
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
            alphabet=self.root_polyclonal.alphabet,
            epitope_colors=self.root_polyclonal.epitope_colors,
            stat="escape_mean",
            error_stat="escape_std",
            **kwargs,
        )

    def mut_escape_site_summary_df_replicates(self, **kwargs):
        """Site-level summaries of mutation escape for replicates.

        Parameters
        ----------
        **kwargs
            Keyword arguments to
            :math:`~polyclonal.polyclonal.Polyclonal.mut_escape_site_summary_df`.
            Note that `min_times_seen` refers to times seen in full data set,
            not individual bootstrap replicates.

        Returns
        -------
        pandas.DataFrame

        """
        if "min_times_seen" in kwargs:
            whitelist = {
                mut
                for (mut, n) in self.root_polyclonal.mutations_times_seen.items()
                if n >= kwargs["min_times_seen"]
            }
            del kwargs["min_times_seen"]
            if "mutation_whitelist" in kwargs:
                kwargs["mutation_whitelist"] = whitelist.union(kwargs["mutation_list"])
            else:
                kwargs["mutation_whitelist"] = whitelist
        return pd.concat(
            [
                m.mut_escape_site_summary_df(**kwargs)
                .assign(bootstrap_replicate=i)
                .drop(columns="n mutations")
                for i, m in enumerate(self.models, start=1)
                if m is not None
            ],
            ignore_index=True,
        )

    def mut_escape_site_summary_df(self, **kwargs):
        """Site-level summaries of mutation escape across replicates.

        Parameters
        ----------
        **kwargs
            Keyword arguments to
            :math:`~polyclonal.polyclonal.Polyclonal.mut_escape_site_summary_df`.
            Note that `min_times_seen` refers to times seen in full data set,
            not individual bootstrap replicates.

        Returns
        -------
        pandas.DataFrame
            The different site-summary metrics ('mean', 'total positive', etc) are
            in different rows for each site and epitope. The 'frac_bootstrap_replicates'
            columns refer to bootstrap replicates with measurements for any mutation
            at that site.

        """
        n_fit = sum(m is not None for m in self.models)
        return (
            self.mut_escape_site_summary_df_replicates(**kwargs)
            .melt(
                id_vars=["epitope", "site", "wildtype", "bootstrap_replicate"],
                var_name="metric",
                value_name="escape",
            )
            .groupby(["epitope", "site", "wildtype", "metric"], as_index=False)
            .aggregate(
                escape_mean=pd.NamedAgg("escape", "mean"),
                escape_std=pd.NamedAgg("escape", "std"),
                n_bootstrap_replicates=pd.NamedAgg("bootstrap_replicate", "count"),
            )
            .assign(
                frac_bootstrap_replicates=lambda x: x["n_bootstrap_replicates"] / n_fit,
            )
            .drop(columns="n_bootstrap_replicates")
            .merge(
                self.root_polyclonal.mut_escape_site_summary_df(**kwargs)[
                    ["site", "n mutations"]
                ].drop_duplicates(),
                on="site",
                how="left",
                validate="many_to_one",
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
            bootstrapped_data=True,
            epitope_colors=self.root_polyclonal.epitope_colors,
            **mut_escape_lineplot_kwargs,
        )

    def icXX_replicates(self, variants_df, **kwargs):
        """Concentration at which a given fraction is neutralized (eg, IC50) for
        all replicates.

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
            and ``bootstrap_replicate`` containing model replicate number.
            Variants with a mutation lacking in a particular bootstrapped model are
            missing in that row.

        """
        return pd.concat(
            [
                m.icXX(m.filter_variants_by_seen_muts(variants_df), **kwargs).assign(
                    bootstrap_replicate=i
                )
                for i, m in enumerate(self.models, start=1)
                if m is not None
            ],
            ignore_index=True,
        )

    def icXX(self, variants_df, **kwargs):
        """Summary statistics of the predicted concentration at which a given
        fraction is neutralized across all replicates.

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
        return (
            self.icXX_replicates(variants_df, **kwargs)
            .groupby(variants_df.columns.tolist(), as_index=False)
            .aggregate(
                mean_IC=pd.NamedAgg(kwargs["col"], "mean"),
                median_IC=pd.NamedAgg(kwargs["col"], "median"),
                std_IC=pd.NamedAgg(kwargs["col"], "std"),
                n_bootstrap_replicates=pd.NamedAgg("bootstrap_replicate", "nunique"),
            )
            .assign(
                frac_bootstrap_replicates=lambda x: x["n_bootstrap_replicates"] / n_fit,
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
            bootstrap replicate. Variants with a mutation lacking in a particular
            bootstrapped model are missing in that row.

        """
        return pd.concat(
            [
                m.prob_escape(
                    variants_df=m.filter_variants_by_seen_muts(variants_df),
                    **kwargs,
                ).assign(bootstrap_replicate=i)
                for i, m in enumerate(self.models, start=1)
                if m is not None
            ],
            ignore_index=True,
        )

    def prob_escape(self, variants_df, **kwargs):
        r"""Compute summary statistics for predicted probability of escape across
        all replicate models.

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
            for each variant at each concentration across bootstrap replicates.

        """
        n_fit = sum(m is not None for m in self.models)
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
                n_bootstrap_replicates=pd.NamedAgg("bootstrap_replicate", "nunique"),
            )
            .assign(
                frac_bootstrap_replicates=lambda x: x["n_bootstrap_replicates"] / n_fit,
            )
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
