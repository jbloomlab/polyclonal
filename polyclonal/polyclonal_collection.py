"""
======================
polyclonal_collection
======================

Defines :class:`PolyclonalCollection` for handling collections of multiple
:mod:`~polyclonal.polyclonal.Polyclonal` objects.

:class:`PolyclonalCollection` is a base class for the following specific use-case
classes:

 - :class:`PolyclonalAverage` for averaging several Polyclonal objects.

 - :class:`PolyclonalBootstrap` for bootstrapping a model.

"""

import copy
import math
import multiprocessing
from functools import partial
from itertools import repeat

import numpy

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
    ----------
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
    ----------
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
        spatial_distances=root_polyclonal.spatial_distances,
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
    ----------
    models_df : pandas.DataFrame
        Data frame of models. Should have one column named "model" that has
        :class:`~polyclonal.polyclonal.Polyclonal` models, and other columns
        are descriptor for model (e.g., "replicate", etc). The descriptors
        for each row must be unique.
    default_avg_to_plot : {"mean", "median"}
        By default when plotting, plot either "mean" or "median".

    Attributes
    ----------
    models : list
        List of the models in `models_df`. All models must have same epitopes.
    model_descriptors : dict
        A list of same length as `models` with each entry being a dict keyed
        by descriptors and values being the descriptor for that model. All models
        must have same descriptor labels. Eg, ``[{"replicate": 1}, {"replicate": 2}]```.
        The descriptor labels are all columns in `models_df` except one named "model".
    descriptor_names : list
        The names that key the entries in :attr:`PolyclonalCollection.model_descriptors`.
    unique_descriptor_names : list
        Names of descriptors in :attr:`PolyclonalCollection.descriptor_names` that are
        not shared across all models.
    epitopes : tuple
        Same meaning as for :attr:`~polyclonal.polyclonal.Polyclonal.epitope`,
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
        self.unique_descriptor_names = [
            name
            for name in self.descriptor_names
            if descriptors_df[name].nunique(dropna=False) > 1
        ]
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
            "epitope",
            as_index=False,
            sort=False,
        ).aggregate(
            activity_mean=pd.NamedAgg("activity", "mean"),
            activity_median=pd.NamedAgg("activity", "median"),
            activity_std=pd.NamedAgg("activity", "std"),
        )

    @property
    def hill_coefficient_df_replicates(self):
        """pandas.DataFrame: Hill coefficients for all models."""
        return pd.concat(
            [
                m.hill_coefficient_df.assign(**desc)
                for m, desc in zip(self.models, self.model_descriptors)
            ],
            ignore_index=True,
        )

    @property
    def hill_coefficient_df(self):
        """pandas.DataFrame: Hill coefficients summarized across models."""
        return self.hill_coefficient_df_replicates.groupby(
            "epitope",
            as_index=False,
            sort=False,
        ).aggregate(
            hill_coefficient_mean=pd.NamedAgg("hill_coefficient", "mean"),
            hill_coefficient_median=pd.NamedAgg("hill_coefficient", "median"),
            hill_coefficient_std=pd.NamedAgg("hill_coefficient", "std"),
        )

    @property
    def non_neutralized_frac_df_replicates(self):
        """pandas.DataFrame: non-neutralizable fraction for all models."""
        return pd.concat(
            [
                m.non_neutralized_frac_df.assign(**desc)
                for m, desc in zip(self.models, self.model_descriptors)
            ],
            ignore_index=True,
        )

    @property
    def non_neutralized_frac_df(self):
        """pandas.DataFrame: non-neutralizable fraction summarized across models."""
        return self.non_neutralized_frac_df_replicates.groupby(
            "epitope",
            as_index=False,
            sort=False,
        ).aggregate(
            non_neutralized_frac_mean=pd.NamedAgg("non_neutralized_frac", "mean"),
            non_neutralized_frac_median=pd.NamedAgg("non_neutralized_frac", "median"),
            non_neutralized_frac_std=pd.NamedAgg("non_neutralized_frac", "std"),
        )

    @property
    def curve_specs_df_replicates(self):
        """pandas.DataFrame: activities, Hill coefficients, and non-neutralized fracs.

        Per-replicate values.

        """
        return pd.concat(
            [
                m.curve_specs_df.assign(**desc)
                for m, desc in zip(self.models, self.model_descriptors)
            ],
            ignore_index=True,
        )

    @property
    def curve_specs_df(self):
        """pandas.DataFrame: activities, Hill coefficients, and non-neutralized fracs.

        Values summarized across models.

        """
        return self.curve_specs_df_replicates.groupby(
            "epitope",
            as_index=False,
            sort=False,
        ).aggregate(
            **{
                f"{param}_{stat}": pd.NamedAgg(param, stat)
                for param in ["activity", "hill_coefficient", "non_neutralized_frac"]
                for stat in ["mean", "median", "std"]
            }
        )

    def curves_plot(self, *, avg_type=None, per_model_lines=5, **kwargs):
        r"""Plot neutralization / binding curve for unmutated protein at each epitope.

        This curve effectively illustrates the epitope activity, Hill curve coefficient,
        and non-neutralizable fraction.

        Parameters
        ----------
        avg_type : {"mean", "median", None}
            Type of average to plot, `None` defaults to
            :attr:`PolyclonalCollection.default_avg_to_plot`.
        per_model_lines : int
            Do we plot thin lines for each model, or just the average? If the number
            of models in the collection is <= than this number, then we plot per-model
            lines, otherwise we just plot the average. A value of -1 means we always plot
            per-model lines.
        **kwargs
            Keywords args for :func:`polyclonal.plot.curves_plot`

        Returns
        -------
        altair.Chart
            Interactive plot.

        """
        if avg_type is None:
            avg_type = self.default_avg_to_plot
        params = ["activity", "hill_coefficient", "non_neutralized_frac"]

        df = self.curve_specs_df.rename(
            columns={f"{c}_{avg_type}": c for c in params}
        ).drop(
            columns=[f"{c}_{s}" for c in params for s in ["mean", "median"]],
            errors="ignore",
        )

        if per_model_lines >= len(self.models) or per_model_lines == -1:
            addtl_tooltip_cols = ["model_name"]
            replicate_col = "model_name"
            df = pd.concat(
                [
                    self.curve_specs_df_replicates.assign(
                        model_name=lambda x: (
                            x[self.unique_descriptor_names]
                            .astype(str)
                            .agg(" ".join, axis=1)
                        )
                    ),
                    df.assign(model_name=avg_type),
                ]
            )[["model_name", "epitope", *params]]
            if len(df) != len(df[["model_name", "epitope"]].drop_duplicates()):
                raise ValueError(f"duplicated model name:\n{df=}")
        else:
            addtl_tooltip_cols = [f"{c}_std" for c in params]
            replicate_col = None

        return polyclonal.plot.curves_plot(
            df,
            "epitope",
            addtl_tooltip_cols=addtl_tooltip_cols,
            names_to_colors=self.epitope_colors,
            replicate_col=replicate_col,
            weighted_replicates=[avg_type],
            **kwargs,
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

    def mut_icXX_df_replicates(self, **kwargs):
        """Get data frame of ICXX and log fold change for each mutation by model.

        Parameters
        ----------
        **kwargs
            Keyword arguments to :meth:`~polyclonal.polyclonal.Polyclonal.mut_icXX_df`

        Returns
        -------
        pandas.DataFrame
            Data from of ICXX and log fold change for each model.

        """
        return pd.concat(
            [
                m.mut_icXX_df(**kwargs).assign(**desc)
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
            "escape_min_magnitude": pd.NamedAgg(
                "escape",
                lambda s: s.tolist()[numpy.argmin(s.abs())],
            ),
            "escape_std": pd.NamedAgg("escape", "std"),
            "n_models": pd.NamedAgg("escape", "count"),
        }
        if "times_seen" in self.mut_escape_df_replicates.columns:
            aggs["times_seen"] = pd.NamedAgg("times_seen", "mean")
        return (
            self.mut_escape_df_replicates.groupby(
                ["epitope", "site", "wildtype", "mutant", "mutation"], as_index=False
            )
            .aggregate(**aggs)
            .assign(
                frac_models=lambda x: x["n_models"] / len(self.models),
                # make categorical to sort, then return to original type
                epitope=lambda x: pd.Categorical(
                    x["epitope"],
                    self.epitopes,
                    ordered=True,
                ),
                site=lambda x: pd.Categorical(
                    x["site"],
                    None if self.sequential_integer_sites else self.sites,
                    ordered=None if self.sequential_integer_sites else True,
                ),
                mutant=lambda x: pd.Categorical(
                    x["mutant"],
                    self.alphabet,
                    ordered=True,
                ),
            )
            .sort_values(["epitope", "site", "mutant"])
            .reset_index(drop=True)
            .assign(
                epitope=lambda x: x["epitope"].tolist(),
                site=lambda x: x["site"].tolist(),
                mutant=lambda x: x["mutant"].tolist(),
            )
        )

    def mut_icXX_df(self, **kwargs):
        """Get data frame of log fold change ICXX induced by each mutation.

        Parameters
        ----------
        **kwargs
            Keyword arguments to :meth:`~polyclonal.polyclonal.Polyclonal.mut_icXX_df`

        Returns
        -------
        pandas.DataFrame
            Log fold change ICXX for each mutation.

        """
        df = self.mut_icXX_df_replicates(**kwargs)
        log_fc_icXX_col = kwargs["log_fold_change_icXX_col"]
        aggs = {
            f"{log_fc_icXX_col} mean": pd.NamedAgg(log_fc_icXX_col, "mean"),
            f"{log_fc_icXX_col} median": pd.NamedAgg(log_fc_icXX_col, "median"),
            f"{log_fc_icXX_col} min_magnitude": pd.NamedAgg(
                log_fc_icXX_col,
                lambda s: s.tolist()[numpy.argmin(s.abs())],
            ),
            f"{log_fc_icXX_col} std": pd.NamedAgg(log_fc_icXX_col, "std"),
            "n_models": pd.NamedAgg(log_fc_icXX_col, "count"),
        }
        if "times_seen" in df.columns:
            aggs["times_seen"] = pd.NamedAgg("times_seen", "mean")
        return (
            df.groupby(["site", "wildtype", "mutant"], as_index=False)
            .aggregate(**aggs)
            .assign(
                frac_models=lambda x: x["n_models"] / len(self.models),
                # make categorical to sort, then return to original type
                site=lambda x: pd.Categorical(
                    x["site"],
                    None if self.sequential_integer_sites else self.sites,
                    ordered=None if self.sequential_integer_sites else True,
                ),
                mutant=lambda x: pd.Categorical(
                    x["mutant"],
                    self.alphabet,
                    ordered=True,
                ),
            )
            .sort_values(["site", "mutant"])
            .reset_index(drop=True)
            .assign(
                site=lambda x: x["site"].tolist(),
                mutant=lambda x: x["mutant"].tolist(),
            )
        )

    @property
    def mut_escape_df_w_model_values(self):
        """pandas.DataFrame: Summarized mutation escape plus per model values.

        Like :attr:`PolyclonalCollection.mut_escape_df` but then having additional
        columns giving per-model escape.

        """
        merge_cols = ["epitope", "site", "wildtype", "mutant", "mutation"]
        return self.mut_escape_df.merge(
            (
                self.mut_escape_df_replicates.assign(
                    model_name=lambda x: (
                        x[self.unique_descriptor_names]
                        .astype(str)
                        .agg(
                            " ".join,
                            axis=1,
                        )
                    ),
                )
                .pivot_table(index=merge_cols, values="escape", columns="model_name")
                .reset_index()
            ),
            on=merge_cols,
            validate="one_to_one",
        )

    def mut_icXX_df_w_model_values(self, **kwargs):
        """Log fold change ICXX induced by each mutation, plus per-model values.

        Like :attr:`PolyclonalCollection.mut_icXX_df` but then having additional
        columns giving per-model ICXXs.

        Parameters
        ----------
        **kwargs
            Keyword arguments to :meth:`~polyclonal.polyclonal.Polyclonal.mut_icXX_df`

        Returns
        -------
        pandas.DataFrame
            Log fold change ICXX for each mutation, plus per model values.

        """
        merge_cols = ["site", "wildtype", "mutant"]
        return self.mut_icXX_df(**kwargs).merge(
            (
                self.mut_icXX_df_replicates(**kwargs)
                .assign(
                    model_name=lambda x: (
                        x[self.unique_descriptor_names]
                        .astype(str)
                        .agg(
                            " ".join,
                            axis=1,
                        )
                    ),
                )
                .pivot_table(
                    index=merge_cols,
                    values=kwargs["log_fold_change_icXX_col"],
                    columns="model_name",
                )
                .reset_index()
            ),
            on=merge_cols,
            validate="one_to_one",
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
            Keyword args for :func:`polyclonal.plot.corr_heatmap`
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

    def mut_escape_plot(
        self,
        *,
        biochem_order_aas=True,
        avg_type=None,
        init_n_models=None,
        prefix_epitope=None,
        df_to_merge=None,
        per_model_tooltip=None,
        **kwargs,
    ):
        """Make plot of mutation escape values.

        Parameters
        ----------
        biochem_order_aas : bool
            Biochemically order amino-acid alphabet :attr:`PolyclonalCollection.alphabet`
            by passing it through :func:`polyclonal.alphabets.biochem_order_aas`.
        avg_type : {"mean", "median", "min_magnitude", None}
            Type of average to plot, None defaults to
            :attr:`PolyclonalCollection.default_avg_to_plot`.
        init_n_models : None or int
            Initially only show mutations found in at least this number of models
            in the collection. A value of `None` corresponds to choosing a
            value that is >= half the number of total replicates.
        prefix_epitope : bool or None
            Do we add the prefix "epitope " to the epitope labels? If `None`, do
            only if epitope is integer.
        df_to_merge : None or pandas.DataFrame or list
            To include additional properties, specify data frame or list of them which
            are merged with :attr:`Polyclonal.mut_escape_df` before being passed
            to :func:`polyclonal.plot.lineplot_and_heatmap`. Properties will
            only be included in plot if relevant columns are passed to
            :func:`polyclonal.plot.lineplot_and_heatmap` via `addtl_slider_stats`,
            `addtl_tooltip_stats`, or `site_zoom_bar_color_col`.
        per_model_tooltip : None or bool
            In the heatmap, do the tooltips report per-model escape values or the
            standard deviation across models. If `None` then report per-model
            when <= 5 models and standard deviation if > 5 models. If `True`,
            always report per-model values. If `False`, always report standard
            deviation.
        **kwargs
            Keyword args for :func:`polyclonal.plot.lineplot_and_heatmap`

        Returns
        -------
        altair.Chart
            Interactive heat maps and line plots.

        """
        if avg_type is None:
            avg_type = self.default_avg_to_plot

        if per_model_tooltip is None:
            per_model_tooltip = len(self.models) <= 5

        if "addtl_tooltip_stats" not in kwargs:
            kwargs["addtl_tooltip_stats"] = []

        if per_model_tooltip:
            df = self.mut_escape_df_w_model_values
            model_names = df.columns[-len(self.models) :]
            for name in model_names:
                if name not in kwargs["addtl_tooltip_stats"]:
                    kwargs["addtl_tooltip_stats"].append(name)
        else:
            df = self.mut_escape_df
            if "escape_std" not in kwargs["addtl_tooltip_stats"]:
                kwargs["addtl_tooltip_stats"].append("escape_std")

        kwargs["data_df"] = pd.concat(
            [
                df.rename(columns={f"escape_{avg_type}": "escape"}),
                (
                    df[["site", "wildtype", "epitope"]]
                    .drop_duplicates()
                    .assign(
                        escape=0,
                        mutant=lambda x: x["wildtype"],
                        **(
                            {name: 0 for name in model_names}
                            if per_model_tooltip
                            else {}
                        ),
                    )
                ),
            ],
        )

        if df_to_merge is not None:
            if isinstance(df_to_merge, pd.DataFrame):
                df_to_merge = [df_to_merge]
            elif not isinstance(df_to_merge, list):
                raise ValueError("`df_to_merge` must be pandas.DataFrame or list")
            for df in df_to_merge:
                if not self.sequential_integer_sites and "site" in df.columns:
                    df = df.assign(site=lambda x: x["site"].astype(str))
                kwargs["data_df"] = kwargs["data_df"].merge(
                    df,
                    how="left",
                    validate="many_to_one",
                )

        if "category_colors" not in kwargs:
            kwargs["category_colors"] = self.epitope_colors

        if prefix_epitope or (
            prefix_epitope is None
            and all(type(e) == int or e.isnumeric() for e in self.epitopes)
        ):
            prefixed = {e: f"epitope {e}" for e in self.epitopes}
            kwargs["data_df"]["epitope"] = kwargs["data_df"]["epitope"].map(prefixed)
            kwargs["category_colors"] = {
                prefixed[e]: color for e, color in kwargs["category_colors"].items()
            }

        kwargs["stat_col"] = "escape"
        kwargs["category_col"] = "epitope"

        if "times_seen" in self.mut_escape_df.columns:
            if "addtl_slider_stats" in kwargs:
                if "times_seen" not in kwargs["addtl_slider_stats"]:
                    kwargs["addtl_slider_stats"]["times_seen"] = 1
            else:
                kwargs["addtl_slider_stats"] = {"times_seen": 1}

        if ("sites" not in kwargs) and not self.sequential_integer_sites:
            kwargs["sites"] = self.sites

        if "alphabet" not in kwargs:
            kwargs["alphabet"] = self.alphabet
        if biochem_order_aas:
            kwargs["alphabet"] = polyclonal.alphabets.biochem_order_aas(
                kwargs["alphabet"]
            )

        if init_n_models is None:
            init_n_models = int(math.ceil(len(self.models) / 2))
        if "addtl_slider_stats" in kwargs:
            kwargs["addtl_slider_stats"]["n_models"] = init_n_models
        else:
            kwargs["addtl_slider_stats"] = {"n_models": init_n_models}

        return polyclonal.plot.lineplot_and_heatmap(**kwargs)

    def mut_icXX_plot(
        self,
        *,
        x=0.9,
        icXX_col="IC90",
        log_fold_change_icXX_col="log2 fold change IC90",
        min_c=1e-5,
        max_c=1e5,
        logbase=2,
        check_wt_icXX=(0.01, 100),
        biochem_order_aas=True,
        df_to_merge=None,
        positive_color=polyclonal.plot.DEFAULT_POSITIVE_COLORS[0],
        negative_color=polyclonal.plot.DEFAULT_NEGATIVE_COLOR,
        avg_type=None,
        init_n_models=None,
        per_model_tooltip=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        x : float
            Same meaning as for :meth:`Polyclonal.mut_icXX_df`.
        icXX_col : str
            Same meaning as for :meth:`Polyclonal.mut_icXX_df`.
        log_fold_change_icXX_col : str
            Same meaning as for :meth:`Polyclonal.mut_icXX_df`.
        min_c : float
            Same meaning as for :meth:`Polyclonal.mut_icXX_df`.
        max_c : float
            Same meaning as for :meth:`Polyclonal.mut_icXX_df`.
        logbase : float
            Same meaning as for :meth:`Polyclonal.mut_icXX_df`.
        check_wt_icXX : None or 2-tuple
            Same meaning as for :meth:`Polyclonal.mut_icXX_df`.
        biochem_order_aas : bool
            Biochemically order the amino-acid alphabet in :attr:`Polyclonal.alphabet`
            by passing it through :func:`polyclonal.alphabets.biochem_order_aas`.
        df_to_merge : None or pandas.DataFrame or list
            To include additional properties, specify data frame or list of them which
            are merged with :attr:`Polyclonal.mut_escape_df` before being passed
            to :func:`polyclonal.plot.lineplot_and_heatmap`. Properties will
            only be included in plot if relevant columns are passed to
            :func:`polyclonal.plot.lineplot_and_heatmap` via `addtl_slider_stats`,
            `addtl_tooltip_stats`, or `site_zoom_bar_color_col`.
        positive_color : str
            Color for positive log fold change in heatmap.
        negative_color : str
            Color for negative log fold change in heatmap.
        avg_type : {"mean", "median", "min_magnitude", None}
            Type of average to plot, None defaults to
            :attr:`PolyclonalCollection.default_avg_to_plot`.
        init_n_models : None or int
            Initially only show mutations found in at least this number of models
            in the collection. A value of `None` corresponds to choosing a
            value that is >= half the number of total replicates.
        per_model_tooltip : None or bool
            In the heatmap, do the tooltips report per-model escape values or the
            standard deviation across models. If `None` then report per-model
            when <= 5 models and standard deviation if > 5 models. If `True`,
            always report per-model values. If `False`, always report standard
            deviation.
        **kwargs
            Keyword args for :func:`polyclonal.plot.lineplot_and_heatmap`

        Returns
        -------
        altair.Chart
            Interactive heat map and line plot.

        """
        if avg_type is None:
            avg_type = self.default_avg_to_plot

        if per_model_tooltip is None:
            per_model_tooltip = len(self.models) <= 5

        if "addtl_tooltip_stats" not in kwargs:
            kwargs["addtl_tooltip_stats"] = []

        if per_model_tooltip:
            df = self.mut_icXX_df_w_model_values(
                x=x,
                icXX_col=icXX_col,
                log_fold_change_icXX_col=log_fold_change_icXX_col,
                min_c=min_c,
                max_c=max_c,
                logbase=logbase,
                check_wt_icXX=check_wt_icXX,
            )
            model_names = df.columns[-len(self.models) :]
            for name in model_names:
                if name not in kwargs["addtl_tooltip_stats"]:
                    kwargs["addtl_tooltip_stats"].append(name)
        else:
            df = self.mut_icXX_df(
                x=x,
                icXX_col=icXX_col,
                log_fold_change_icXX_col=log_fold_change_icXX_col,
                min_c=min_c,
                max_c=max_c,
                logbase=logbase,
                check_wt_icXX=check_wt_icXX,
            )
            if "escape_std" not in kwargs["addtl_tooltip_stats"]:
                kwargs["addtl_tooltip_stats"].append("escape_std")

        kwargs["data_df"] = df.assign(epitope="all").rename(
            columns={f"{log_fold_change_icXX_col} {avg_type}": log_fold_change_icXX_col}
        )

        if init_n_models is None:
            init_n_models = int(math.ceil(len(self.models) / 2))
        if "addtl_slider_stats" in kwargs:
            kwargs["addtl_slider_stats"]["n_models"] = init_n_models
        else:
            kwargs["addtl_slider_stats"] = {"n_models": init_n_models}

        if df_to_merge is not None:
            if isinstance(df_to_merge, pd.DataFrame):
                df_to_merge = [df_to_merge]
            elif not isinstance(df_to_merge, list):
                raise ValueError("`df_to_merge` must be pandas.DataFrame or list")
            for df in df_to_merge:
                if not self.sequential_integer_sites and "site" in df.columns:
                    df = df.assign(site=lambda x: x["site"].astype(str))
                kwargs["data_df"] = kwargs["data_df"].merge(
                    df,
                    how="left",
                    validate="many_to_one",
                )

        kwargs["category_colors"] = {"all": positive_color}
        kwargs["heatmap_negative_color"] = negative_color

        kwargs["stat_col"] = log_fold_change_icXX_col
        kwargs["category_col"] = "epitope"

        if "times_seen" in kwargs["data_df"].columns:
            if "times_seen" not in kwargs["addtl_slider_stats"]:
                kwargs["addtl_slider_stats"]["times_seen"] = 1

        if "init_floor_at_zero" not in kwargs:
            kwargs["init_floor_at_zero"] = False

        if "heatmap_min_at_least" not in kwargs:
            kwargs["heatmap_min_at_least"] = -2
        if "heatmap_max_at_least" not in kwargs:
            kwargs["heatmap_max_at_least"] = 2

        if ("sites" not in kwargs) and not self.sequential_integer_sites:
            kwargs["sites"] = self.sites

        if "alphabet" not in kwargs:
            kwargs["alphabet"] = self.alphabet
        if biochem_order_aas:
            kwargs["alphabet"] = polyclonal.alphabets.biochem_order_aas(
                kwargs["alphabet"]
            )

        return polyclonal.plot.lineplot_and_heatmap(**kwargs)

    def mut_escape_site_summary_df_replicates(self, **kwargs):
        """Site-level summaries of mutation escape for models.

        Parameters
        ----------
        **kwargs
            Keyword arguments to
            :meth:`~polyclonal.polyclonal.Polyclonal.mut_escape_site_summary_df`.

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
                ["epitope", "site", "wildtype", "metric"],
                as_index=False,
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
            # make categorical to sort, then unsort
            .assign(
                epitope=lambda x: pd.Categorical(
                    x["epitope"],
                    self.epitopes,
                    ordered=True,
                ),
                site=lambda x: pd.Categorical(
                    x["site"],
                    None if self.sequential_integer_sites else self.sites,
                    ordered=None if self.sequential_integer_sites else True,
                ),
            )
            .sort_values(["epitope", "site"])
            .reset_index(drop=True)
            .assign(
                epitope=lambda x: x["epitope"].tolist(),
                site=lambda x: x["site"].tolist(),
            )
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
    ----------
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
    ----------
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
