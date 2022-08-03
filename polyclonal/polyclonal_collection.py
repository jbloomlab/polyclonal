"""
======================
polyclonal_collection
======================

Defines :class:`PolyclonalCollection` for handling collections of multiple
:mod:`~polyclonal.polyclonal.Polyclonal` objects.

:class:`PolyclonalCollection` is a base class for the following specific use-case
classes:

 - :class:`PolyclonalAverage` for bootstrapping a model.

 - :class:`PolyclonalBootstrap` for bootstrapping a model.

"""

import copy
import math
import multiprocessing
from functools import partial
from itertools import repeat

import natsort

import pandas as pd

import polyclonal
import polyclonal.alphabets
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
        sites=(
            None if root_polyclonal.sequential_integer_sites else root_polyclonal.sites
        ),
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
    default_avg_to_plot : {"mean", "median"}
        By default when plotting, plot either "mean" or "median".

    Attributes
    -----------
    models : list
        List of the models in `models_df`. All models must have same epitopes.
    model_descriptors : dict
        A list of same length as `models` with each entry being a dict keyed
        by descriptors and values being the descriptor for that model. All models
        must have same descriptor labels. Eg, ``[{"replicate": 1}, {"replicate": 2}]```.
        The descriptor labels are all columns in `models_df` except one named "model".
    descriptor_names : list
        The names that key the entries in :attr:`PolyclonalCollection.model_descriptors`.
    epitopes : tuple
        Same meaning as for :attr:`~polyclonal.polyclonal.Polyclonal.epitope_colors`,
        extracted from :attr:`PolyclonalCollection.models`.
    epitope_colors : dict
        Same meaning as for :attr:`~polyclonal.polyclonal.Polyclonal.epitope_colors`,
        extracted from :attr:`PolyclonalCollection.models`.
    alphabet : array-like
        Same meaning as for :attr:`~polyclonal.polyclonal.Polyclonal.alphabet`,
        extracted from :attr:`PolyclonalCollection.models`.
    sequential_integer_sites : bool
        Same as for :attr:`~polyclonal.polyclonal.Polyclonal.sequential_integer_sites`,
        extracted from :attr:`PolyclonalCollection.models`.
    default_avg_to_plot : {"mean", "median"}
        By default when plotting, plot either "mean" or "median".

    """

    def __init__(self, models_df, *, default_avg_to_plot):
        """See main class docstring for details."""
        if default_avg_to_plot not in {"mean", "median"}:
            raise ValueError(f"invalid {default_avg_to_plot=}")
        self.default_avg_to_plot = default_avg_to_plot

        self.models = models_df["model"].tolist()
        if len(self.models) < 1:
            raise ValueError(f"No models:\n{models_df=}")

        descriptors_df = models_df.drop(columns="model").reset_index(drop=True)
        if not len(descriptors_df.columns):
            raise ValueError("not descriptor columns in `models_df`")
        self.descriptor_names = descriptors_df.columns.tolist()
        if len(descriptors_df.drop_duplicates()) != len(self.models):
            raise ValueError("some models have the same descriptors")
        self.model_descriptors = list(descriptors_df.to_dict(orient="index").values())

        for attr in [
            "epitopes",
            "epitope_colors",
            "alphabet",
            "sequential_integer_sites",
        ]:
            for model in self.models:
                if not hasattr(self, attr):
                    setattr(self, attr, copy.copy(getattr(model, attr)))
                elif getattr(self, attr) != getattr(model, attr):
                    raise ValueError(f"{attr} not the same for all models")
        if not self.models[0].sequential_integer_sites:
            attr = "sites"
            for model in self.models:
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
            activity_median=pd.NamedAgg("activity", "median"),
            activity_std=pd.NamedAgg("activity", "std"),
        )

    def activity_wt_barplot(self, avg_type=None, **kwargs):
        """Bar plot of epitope activities mean across models.

        Parameters
        ----------
        avg_type : {"mean", "median", None}
            Type of average to plot, None defaults to
            :attr:`PolyclonalCollection.default_avg_to_plot`.
        **kwargs
            Keyword arguments for :func:`polyclonal.plot.activity_wt_barplot`.

        Returns
        -------
        altair.Chart
            Interactive plot, with error bars showing standard deviation.

        """
        if avg_type is None:
            avg_type = self.default_avg_to_plot
        return polyclonal.plot.activity_wt_barplot(
            activity_wt_df=self.activity_wt_df,
            epitope_colors=self.epitope_colors,
            stat=f"activity_{avg_type}",
            error_stat="activity_std",
            **kwargs,
        )

    @property
    def mut_escape_df_replicates(self):
        """pandas.DataFrame: Mutation escape by model."""
        return pd.concat(
            [
                m.mut_escape_df.assign(**desc)
                for m, desc in zip(self.models, self.model_descriptors)
            ],
            ignore_index=True,
        )

    @property
    def mut_escape_df(self):
        """pandas.DataFrame: Mutation escape summarized across models."""
        aggs = {
            "escape_mean": pd.NamedAgg("escape", "mean"),
            "escape_median": pd.NamedAgg("escape", "median"),
            "escape_std": pd.NamedAgg("escape", "std"),
            "n_models": pd.NamedAgg("escape", "count"),
        }
        if "times_seen" in self.mut_escape_df_replicates.columns:
            aggs["times_seen"] = pd.NamedAgg("times_seen", "mean")
        return (
            self.mut_escape_df_replicates.groupby(
                ["epitope", "site", "wildtype", "mutant", "mutation"],
                as_index=False,
                sort=False,
            )
            .aggregate(**aggs)
            .assign(
                frac_models=lambda x: x["n_models"] / len(self.models),
            )
        )

    def mut_escape_corr(self, method="pearson"):
        """Correlation of mutation escape values across models for each epitope.

        Parameters
        ----------
        method : str
            A correlation method passable to `pandas.DataFrame.corr`.

        Returns
        -------
        pandas.DataFrame
            Tidy data frame giving correlations between models for all epitopes.
            The models are labeled by their descriptors suffixed with "_1" and
            "_2" for the two models being compared.

        """
        # get unique id for each descriptor
        if "_id" in self.mut_escape_df_replicates.columns:
            raise ValueError("`mut_escape_df_replicates` cannot have column '_id'")
        ids = (
            self.mut_escape_df_replicates[self.descriptor_names]
            .drop_duplicates()
            .reset_index(drop=True)
            .assign(_id=lambda x: x.index + 1)
        )
        assert len(ids) == len(self.models)

        corr = (
            polyclonal.utils.tidy_to_corr(
                self.mut_escape_df_replicates.merge(
                    ids,
                    on=self.descriptor_names,
                    validate="many_to_one",
                ),
                sample_col="_id",
                label_col="mutation",
                value_col="escape",
                group_cols="epitope",
                method=method,
            )
            .merge(ids, left_on="_id_1", right_on="_id", validate="many_to_one")
            .rename(columns={n: f"{n}_1" for n in self.descriptor_names})
            .drop(columns=["_id", "_id_1"])
            .merge(ids, left_on="_id_2", right_on="_id", validate="many_to_one")
            .rename(columns={n: f"{n}_2" for n in self.descriptor_names})
            .drop(columns=["_id", "_id_2"])
        )

        return corr

    def mut_escape_corr_heatmap(self, method="pearson", plot_corr2=True, **kwargs):
        """Heatmap of mutation-escape correlation among models at each epitope.

        Parameters
        ----------
        method : str
            A correlation method passable to `pandas.DataFrame.corr`.
        plot_corr2 : bool
            Plot squared correlation (eg, :math:`R^2` rather :math:`R`).
        **kwargs
            Keyword args for :func:`polyclonal.plot.mut_escape_heatmap`
        """
        corr_label = {"pearson": "r", "kendall": "tau", "spearman": "rho"}[method]
        corr2_label = f"{corr_label}2"
        corr_df = (
            self.mut_escape_corr(method)
            .assign(correlation2=lambda x: x["correlation"] ** 2)
            .rename(columns={"correlation": corr_label, "correlation2": corr2_label})
        )
        corr_df = corr_df[
            [corr_label, corr2_label]
            + [c for c in corr_df.columns if c not in {corr_label, corr2_label}]
        ]

        if "corr_range" not in kwargs:
            if plot_corr2:
                kwargs["corr_range"] = (0, 1)
            else:
                min_corr = corr_df[corr_label].min()
                kwargs["corr_range"] = (-1 if min_corr < 0 else -1, 1)

        return polyclonal.plot.corr_heatmap(
            corr_df=corr_df,
            corr_col=corr2_label if plot_corr2 else corr_label,
            sample_cols=self.descriptor_names,
            group_col=None if corr_df["epitope"].nunique() == 1 else "epitope",
            **kwargs,
        )

    def mut_escape_heatmap(
        self,
        *,
        biochem_order_aas=True,
        avg_type=None,
        init_n_replicates=None,
        **kwargs,
    ):
        """Heatmaps of mutation escape values.

        Parameters
        ----------
        biochem_order_aas : bool
            Biochemically order amino-acid alphabet :attr:`PolyclonalCollection.alphabet`
            by passing it through :func:`polyclonal.alphabets.biochem_order_aas`.
        avg_type : {"mean", "median", None}
            Type of average to plot, None defaults to
            :attr:`PolyclonalCollection.default_avg_to_plot`.
        init_n_replicates : None or int
            Initially only show mutations found in at least this number of replicates
            (models in the collection). A value of `None` corresponds to choosing a
            value that is >= half the number of total replicates.
        **kwargs
            Keyword args for :func:`polyclonal.plot.mut_escape_heatmap`

        Returns
        -------
        altair.Chart
            Interactive heat maps.

        """
        if "addtl_tooltip_stats" not in kwargs:
            if "times_seen" in self.mut_escape_df.columns:
                kwargs["addtl_tooltip_stats"] = ["times_seen"]

        if "sites" not in kwargs and not self.sequential_integer_sites:
            kwargs["sites"] = self.sites
        if init_n_replicates is None:
            init_n_replicates = int(math.ceil(len(self.models) / 2))
        if (
            "addtl_tooltip_stats" in kwargs
            and kwargs["addtl_tooltip_stats"] is not None
        ):
            kwargs["addtl_tooltip_stats"].append("n_replicates")
        else:
            kwargs["addtl_tooltip_stats"] = ["n_replicates"]
        if "addtl_slider_stats" in kwargs and kwargs["addtl_slider_stats"] is not None:
            kwargs["addtl_slider_stats"]["n_replicates"] = init_n_replicates
        else:
            kwargs["addtl_slider_stats"] = {"n_replicates": init_n_replicates}

        if avg_type is None:
            avg_type = self.default_avg_to_plot

        return polyclonal.plot.mut_escape_heatmap(
            mut_escape_df=self.mut_escape_df.rename(
                columns={"n_models": "n_replicates"}
            ),
            alphabet=(
                polyclonal.alphabets.biochem_order_aas(self.alphabet)
                if biochem_order_aas
                else self.alphabet
            ),
            epitope_colors=self.epitope_colors,
            stat=f"escape_{avg_type}",
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
                for m, desc in zip(self.models, self.model_descriptors)
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
            In particular, you may want to use `min_times_seen`.

        Returns
        -------
        pandas.DataFrame
            The different site-summary metrics ('mean', 'total positive', etc) are
            in different rows for each site and epitope. The 'frac_models'
            column refers to models with measurements for any mutation at that site.

        """
        df = self.mut_escape_site_summary_df_replicates(**kwargs)
        return (
            df.drop(columns="n mutations")
            .melt(
                id_vars=["epitope", "site", "wildtype", *self.descriptor_names],
                var_name="metric",
                value_name="escape",
            )
            .groupby(
                ["epitope", "site", "wildtype", "metric"], as_index=False, sort=False,
            )
            .aggregate(
                escape_mean=pd.NamedAgg("escape", "mean"),
                escape_median=pd.NamedAgg("escape", "median"),
                escape_std=pd.NamedAgg("escape", "std"),
                n_models=pd.NamedAgg("escape", "count"),
            )
            .assign(
                frac_models=lambda x: x["n_models"] / len(self.models),
            )
            .merge(
                df.groupby(["epitope", "site"]).aggregate({"n mutations": "mean"}),
                on=["epitope", "site"],
                validate="many_to_one",
            )
        )

    def mut_escape_lineplot(
        self,
        *,
        avg_type=None,
        min_replicates=None,
        mut_escape_site_summary_df_kwargs=None,
        mut_escape_lineplot_kwargs=None,
    ):
        """Line plots of mutation escape at each site.

        Parameters
        ----------
        avg_type : {"mean", "median", None}
            Type of average to plot, None defaults to
            :attr:`PolyclonalCollection.default_avg_to_plot`.
        min_replicates : None or int
            Only include sites that have escape estimated for at least this many models.
            A value of `None` corresponds to choosing a value that is >= half the number
            of total replicates.
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
        if min_replicates is None:
            min_replicates = int(math.ceil(len(self.models) / 2))
        if mut_escape_site_summary_df_kwargs is None:
            mut_escape_site_summary_df_kwargs = {}
        if mut_escape_lineplot_kwargs is None:
            mut_escape_lineplot_kwargs = {}
        if "avg_to_plot" not in mut_escape_lineplot_kwargs:
            if avg_type is None:
                avg_type = self.default_avg_to_plot
            mut_escape_lineplot_kwargs["avg_to_plot"] = f"escape_{avg_type}"
        if "addtl_tooltip_stats" not in mut_escape_lineplot_kwargs:
            mut_escape_lineplot_kwargs["addtl_tooltip_stats"] = ["n mutations"]
        if (
            "sites" not in mut_escape_lineplot_kwargs
            and not self.sequential_integer_sites
        ):
            mut_escape_lineplot_kwargs["sites"] = self.sites
        df = self.mut_escape_site_summary_df(**mut_escape_site_summary_df_kwargs).query(
            "n_models >= @min_replicates"
        )
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
                for m, desc in zip(self.models, self.model_descriptors)
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
            De-duplicated opy of ``variants_df`` with added column ``col`` containing
            icXX and summary stats for each variant across all models.

        """
        if "col" in kwargs:
            col = kwargs["col"]
        else:
            col = "IC50"
        variants_df = variants_df.drop_duplicates()
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
                frac_models=lambda x: x["n_models"] / len(self.models),
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
                for m, desc in zip(self.models, self.model_descriptors)
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
            De-duplicated copy of ``variants_df`` with columns named 'concentration'
            and 'mean', 'median', and 'std' giving corresponding summary stats
            of predicted probability of escape :math:`p_v\left(c\right)`
            for each variant at each concentration across models.

        """
        variants_df = variants_df.drop_duplicates()
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
                frac_models=lambda x: x["n_models"] / len(self.models),
            )
        )


class PolyclonalAverage(PolyclonalCollection):
    """Average several :class:`~polyclonal.polyclonal.Polyclonal` objects.

    Parameters
    ----------
    models_df : pandas.DataFrame
        Same meaning as for :class:`PolyclonalCollection`. However, the resulting
        collection of models will have **copies** of these models rather than the
        actual objects in `models_df`.
    harmonize_to : :class:`PolyclonalCollection` or None
        When harmonizing the epitopes, harmonize to this model. If `None`, just
        harmonize to the first model in `models_df`.
    default_avg_to_plot : {"mean", "median"}
        What type of average do the plotting methods plot by default?

    Attributes
    ----------
    Other attributes of :class:`PolyclonalCollection`.
        Inherited from base class.

    """

    def __init__(self, models_df, *, harmonize_to=None, default_avg_to_plot="median"):
        """See main class docstring."""
        if not len(models_df):
            raise ValueError("no models in `model_df`")
        if harmonize_to is None:
            harmonize_to = models_df.iloc[0]["model"]

        models_df["model"] = [
            m.epitope_harmonized_model(harmonize_to)[0] for m in models_df["model"]
        ]

        super().__init__(models_df, default_avg_to_plot=default_avg_to_plot)


class PolyclonalBootstrap(PolyclonalCollection):
    """Bootstrap :class:`~polyclonal.polyclonal.Polyclonal` objects.

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
    default_avg_to_plot : {"mean", "median"}
        What type of average do the plotting methods plot by default?

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
        *,
        n_threads=-1,
        seed=0,
        sample_by="barcode",
        default_avg_to_plot="mean",
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
            pd.DataFrame({"model": [m for m in models if m is not None]}).assign(
                bootstrap_replicate=lambda x: x.index + 1
            ),
            default_avg_to_plot=default_avg_to_plot,
        )

    def fit_models(self, failures="error", **kwargs):
        """Fits bootstrapped :class:`~polyclonal.polyclonal.Polyclonal` models.

        The fit models will then be in :attr:`PolyclonalCollection.models`,
        with any models that fail fitting set to `None`. Their epitopes will
        also be harmonized with :attr:`PolyclonalBootstrap.root_polyclonal`.

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

        self.models = [
            None if m is None else m.epitope_harmonized_model(self.root_polyclonal)[0]
            for m in self.models
        ]

        return (n_fit, n_failed)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
