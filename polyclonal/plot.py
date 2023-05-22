"""
==========
plot
==========

Plotting functions.

"""


import functools
import math
import operator

import altair as alt

import matplotlib.colors

import natsort

import numpy

import pandas as pd


alt.data_transformers.disable_max_rows()


TAB10_COLORS_NOGRAY = tuple(
    c for c in matplotlib.colors.TABLEAU_COLORS.values() if c != "#7f7f7f"
)
"""tuple: Tableau 10 color palette without gray."""

DEFAULT_POSITIVE_COLORS = ("#0072B2", "#CC79A7", "#009E73", "#17BECF", "#BCDB22")
"""tuple: Colors in hex: french blue, wild orchid, green, light blue, olive."""

DEFAULT_NEGATIVE_COLOR = "#E69F00"
"""str: Orange from cbPalette color in hex."""


def color_gradient_hex(start, end, n):
    """Get a list of colors linearly spanning a range.

    Parameters
    ----------
    start : str
        Starting color.
    end : str
        Ending color.
    n : int
        Number of colors in list.

    Returns
    -------
    list
        List of hex codes for colors spanning `start` to `end`.

    Example
    -------
    >>> color_gradient_hex('white', 'red', n=5)
    ['#ffffff', '#ffbfbf', '#ff8080', '#ff4040', '#ff0000']

    """
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        name="_", colors=[start, end], N=n
    )
    return [matplotlib.colors.rgb2hex(tup) for tup in cmap(list(range(0, n)))]


def activity_wt_barplot(
    *,
    activity_wt_df,
    epitope_colors,
    epitopes=None,
    stat="activity",
    error_stat=None,
    width=110,
    height_per_bar=25,
):
    r"""Bar plot of activity against each epitope, :math:`a_{\rm{wt},e}`.

    Parameters
    ----------
    activity_wt_df : pandas.DataFrame
        Epitope activities in format of
        :attr:`polyclonal.polyclonal.Polyclonal.activity_wt_df`.
    epitope_colors : dict
        Maps each epitope name to its color.
    epitopes : array-like or None
        Include these epitopes in this order. If `None`, use all epitopes
        in order found in ``activity_wt_df``.
    stat : str
        Statistic in `activity_wt_df` to plot as activity.
    error_stat : str or None
        Statistic in `activity_wt_df` to plot as error for bars.
    width : float
        Width of plot.
    height_per_bar : float
        Height of plot for each bar (epitope).

    Returns
    -------
    altair.Chart
        Interactive plot.

    """
    if epitopes is None:
        epitopes = activity_wt_df["epitope"].tolist()
    elif not set(epitopes).issubset(activity_wt_df["epitope"]):
        raise ValueError("invalid entries in `epitopes`")

    if stat not in activity_wt_df.columns:
        raise ValueError(f"{stat=} not in {activity_wt_df.columns=}")

    if error_stat is not None:
        if error_stat not in activity_wt_df.columns:
            raise ValueError(f"{error_stat=} not in {activity_wt_df.columns=}")
        assert not {"_upper", "_lower"}.intersection(activity_wt_df.columns)
        df = activity_wt_df.assign(
            _lower=lambda x: x[stat] - x[error_stat],
            _upper=lambda x: x[stat] + x[error_stat],
        )
    else:
        df = activity_wt_df

    baseplot = alt.Chart(df).encode(
        y=alt.Y("epitope:N", sort=epitopes),
        color=alt.Color(
            "epitope:N",
            scale=alt.Scale(
                domain=epitopes, range=[epitope_colors[e] for e in epitopes]
            ),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip(c, format=".2f") if df[c].dtype == float else c
            for c in df.columns
            if c not in ["_upper", "_lower"]
        ],
    )

    barplot = baseplot.encode(x=alt.X(f"{stat}:Q", title=stat)).mark_bar(
        size=0.75 * height_per_bar
    )

    if error_stat is not None:
        barplot = barplot + (
            baseplot.encode(
                x="_lower", x2="_upper", color=alt.value("black")
            ).mark_rule(size=2)
        )

    return barplot.configure_axis(grid=False).properties(
        width=width, height={"step": height_per_bar}
    )


def curves_plot(
    curve_specs_df,
    name_col,
    *,
    names_to_colors=None,
    unbound_label="fraction not neutralized",
    npoints=200,
    concentration_range=50,
    height=125,
    width=225,
    addtl_tooltip_cols=None,
    replicate_col=None,
    weighted_replicates=None,
):
    r"""Plot Hill curves.

    The curves are defined by
    :math:`U_e = \frac{1 - t_e}{1 + \left[c \exp \left(a_e\right)\right]^{n_e}} + t_e`
    where :math:`U_e` is the unbound fraction (plotted on y-axis), :math:`c` is the
    concentration (plotted on x-axis), :math:`a_e` is the activity, :math:`n_e` is
    the Hill coefficient, and :math:`t_e` is the non-neutralizable fraction. A different
    plot is made for each curve name (eg, epitope :math:`e`).

    Parameters
    ----------
    curve_specs_df : pandas.DataFrame
        Should have columns `name_col` (giving name, eg epitope), 'activity',
        'hill_coefficient', and 'non_neutralized_frac' specifying each curves.
        curve.
    name_col : pandas.DataFrame
        Name of column in `curve_specs_df` giving the curve name (eg, epitope)
    names_to_colors : dict or None
        To specify colors for each entry in `name_col` (eg, epitope), provide
        dict mapping names to colors.
    unbound_label : str
        Label for the y-axis, :math:`U_e`.
    npoints : int
        Number of points used to calculate the smoothed line that is plotted.
    concentration_range : float or tuple
        If a float, then plot concentrations from this many fold lower than minimum
        :math:`\exp\left(-a_e\right)` to this many folder greater than maximum
        :math:`\excp\left(-a_e\right)`. If a 2-tuple, then plot concentrations in the
        specified fixed range.
    height : float
        Plot height.
    width : float
        Plot width.
    addtl_tooltip_cols : None or list
        Additional columns in `curve_specs_df` to show as tooltips.
    replicate_col : None or str
        If there are multiple replicates with `name_col`, specify column with their names
        here and a line is plotted for each.
    weighted_replicates : None or list
        If you want to plot only some replicates (such as 'mean') with a heavily weighted
        line and the rest with a thinner line, provide list of those to plot with
        heavily weighted line. `None` means all are heavily weighted.

    Returns
    -------
    altair.Chart
        Interactive plot.

    """
    req_cols = {name_col, "activity", "hill_coefficient", "non_neutralized_frac"}
    if replicate_col:
        req_cols.add(replicate_col)
    if not req_cols.issubset(curve_specs_df):
        raise ValueError(f"{curve_specs_df.columns} lacks columns in {req_cols}")
    if len(curve_specs_df) != len(
        curve_specs_df[
            [name_col, replicate_col] if replicate_col else [name_col]
        ].drop_duplicates()
    ):
        raise ValueError(f"{name_col=} not unique in {curve_specs_df=}")

    # get concentrations to plot
    if hasattr(concentration_range, "__len__"):
        min_c, max_c = concentration_range
    else:
        exp_neg_a = numpy.exp(-curve_specs_df["activity"])
        min_c = exp_neg_a.min() / concentration_range
        max_c = exp_neg_a.max() * concentration_range
    if min_c >= max_c:
        raise ValueError(f"invalid concentration range of {min_c} to {max_c=}")
    cs = numpy.logspace(math.log10(min_c), math.log10(max_c), npoints, base=10)

    if {"u", "c"}.intersection(curve_specs_df.columns):
        raise ValueError("`curve_specs_df` cannot have columns 'u' or 'c'")
    df = curve_specs_df.merge(pd.DataFrame({"c": cs}), how="cross").assign(
        u=lambda x: (
            (1 - x["non_neutralized_frac"])
            / (1 + (x["c"] * numpy.exp(x["activity"])) ** x["hill_coefficient"])
            + x["non_neutralized_frac"]
        ),
    )

    addtl_tooltips = []
    if addtl_tooltip_cols:
        for c in addtl_tooltip_cols:
            if c not in curve_specs_df.columns:
                raise ValueError(
                    f"`addtl_tooltip_cols` column {c} not in `curve_specs_df`"
                )
            if curve_specs_df[c].dtype == float:
                addtl_tooltips.append(alt.Tooltip(c, format=".3g"))
            else:
                addtl_tooltips.append(c)

    select_name = alt.selection_point(
        fields=[name_col],
        bind="legend",
    )

    names = curve_specs_df[name_col].unique().tolist()

    # properties of heavy and light lines
    is_weighted_col = "is_weighted_replicate"
    assert is_weighted_col not in df.columns
    if replicate_col:
        if weighted_replicates is None:
            df[is_weighted_col] = 1
        else:
            df[is_weighted_col] = (
                df[replicate_col].isin(weighted_replicates).astype(int)
            )
    else:
        df[is_weighted_col] = 1

    chart = (
        alt.Chart(df)
        .encode(
            x=alt.X(
                "c",
                title="concentration",
                scale=alt.Scale(type="log", nice=False),
            ),
            y=alt.Y("u", title=unbound_label),
            color=alt.Color(
                name_col,
                sort=names,
                scale=(
                    alt.Scale(domain=names, range=[names_to_colors[n] for n in names])
                    if names_to_colors
                    else alt.Scale()
                ),
            ),
            tooltip=[
                name_col,
                alt.Tooltip("activity", format=".3g"),
                alt.Tooltip("hill_coefficient", format=".3g"),
                alt.Tooltip("non_neutralized_frac", format=".3g"),
                alt.Tooltip("c", format=".3g", title="concentration"),
                alt.Tooltip("u", format=".3g", title=unbound_label),
                *addtl_tooltips,
            ],
            strokeWidth=alt.StrokeWidth(
                is_weighted_col,
                scale=alt.Scale(domain=(0, 1), range=(1, 2.5), nice=False),
                legend=None,
            ),
            opacity=alt.Opacity(
                is_weighted_col,
                scale=alt.Scale(domain=(0, 1), range=(0.5, 1), nice=False),
                legend=None,
            ),
            detail=(alt.Detail(replicate_col) if replicate_col else alt.Detail()),
        )
        .mark_line()
        .configure_axis(grid=False, labelOverlap=True)
        .configure_legend(symbolStrokeWidth=3)
        .properties(height=height, width=width)
        .add_params(select_name)
        .transform_filter(select_name)
    )

    return chart


def corr_heatmap(
    corr_df,
    corr_col,
    sample_cols,
    *,
    group_col=None,
    corr_range=(0, 1),
    columns=3,
    diverging_colors=None,
    scheme=None,
):
    """Plot a correlation matrix as heat map from a tidy data frame of correlations.

    Parameters
    ----------
    corr_df : pandas.DataFrame
        Data to plot.
    corr_col : str
        Column in `corr_df` with correlation coefficient.
    sample_cols : str or list
        Column(s) in corresponding to sample identifiers, suffixed by "_1" and "_2" for
        the distinct samples. Should be entries for all pairs of samples.
    group_col : str or None
        Column in `corr_df` to facet plots on, or `None` if no facets.
    corr_range : tuple or None
        Range of heat map as `(min, max)`, or `None` to use data range. Typically
        you will want to set to `(0, 1)` for :math:`r^2` and `(-1, 1)` for :math:`r`.
    columns : int
        Facet by `group_col` into this many columns.
    diverging_colors : None or bool
        If `True`, mid point of color scale is set to zero. If `None`, select `True`
        if `corr_range` extends below 0.
    scheme : None or str
        Color scheme to use, see https://vega.github.io/vega/docs/schemes/.
        If `None`, choose intelligently based on `corr_range` and `diverging_colors`.

    Returns
    -------
    altair.Chart
        Heatmap(s) of correlation coefficients.

    """
    corr_df = corr_df.copy()  # so we don't change input data frame

    if corr_col not in corr_df:
        raise ValueError(f"{corr_col=} not in {corr_df.columns=}")

    if corr_range is None:
        corr_range = (corr_df[corr_col].min(), corr_df[corr_col].max())
    else:
        if corr_range[0] > corr_df[corr_col].min():
            raise ValueError(f"{corr_range[0]} > {corr_df[corr_col].min()=}")
        if corr_range[1] < corr_df[corr_col].max():
            raise ValueError(f"{corr_range[1]} < {corr_df[corr_col].max()=}")

    if diverging_colors is None:
        diverging_colors = corr_range[0] < 0
    if scheme is None:
        scheme = "redblue" if diverging_colors else "yellowgreenblue"

    if (group_col is not None) and group_col not in corr_df:
        raise ValueError(f"{group_col=} not in {corr_df.columns=}")

    # check column exists and make labels by concatenating sample_cols
    if isinstance(sample_cols, str):
        sample_cols = [sample_cols]
    for suffix in ["_1", "_2"]:
        label_col = f"_label{suffix}"
        if label_col in corr_df.columns:
            raise ValueError(f"`corr_df` cannot have column {label_col}")
        label = []
        for col in sample_cols:
            col_suffixed = f"{col}{suffix}"
            if col_suffixed not in corr_df.columns:
                raise ValueError(f"{col_suffixed} not in {corr_df.columns=}")
            else:
                label.append(col_suffixed)
        # https://stackoverflow.com/a/49122979
        labstr = "-".join(["{}"] * len(label))
        corr_df[label_col] = [labstr.format(*r) for r in corr_df[label].values.tolist()]

    corr_chart = (
        alt.Chart(corr_df)
        .encode(
            x=alt.X("_label_1", title=None),
            y=alt.Y("_label_2", title=None),
            color=alt.Color(
                corr_col,
                scale=(
                    alt.Scale(domainMid=0, domain=corr_range, scheme=scheme)
                    if diverging_colors
                    else alt.Scale(domain=corr_range, scheme=scheme)
                ),
            ),
            tooltip=[
                alt.Tooltip(c, format=".3f") if corr_df[c].dtype == float else c
                for c in corr_df.columns
                if c not in {"_label_1", "_label_2"}
            ],
            facet=(
                alt.Facet()
                if group_col is None
                else alt.Facet(group_col, columns=columns)
            ),
        )
        .mark_rect(stroke="black")
        .properties(width=alt.Step(15), height=alt.Step(15))
        .configure_axis(labelLimit=500)
    )

    return corr_chart


def lineplot_and_heatmap(
    *,
    data_df,
    stat_col,
    category_col,
    alphabet=None,
    sites=None,
    addtl_tooltip_stats=None,
    addtl_slider_stats=None,
    addtl_slider_stats_hide_not_filter=None,
    init_floor_at_zero=True,
    init_site_statistic="sum",
    cell_size=11,
    lineplot_width=690,
    lineplot_height=100,
    site_zoom_bar_width=500,
    site_zoom_bar_color_col=None,
    plot_title=None,
    show_single_category_label=False,
    category_colors=None,
    heatmap_negative_color=None,
    heatmap_color_scheme=None,
    heatmap_color_scheme_mid_0=True,
    heatmap_max_at_least=None,
    heatmap_min_at_least=None,
    heatmap_max_fixed=None,
    heatmap_min_fixed=None,
    site_zoom_bar_color_scheme="set3",
    slider_binding_range_kwargs=None,
    hide_color="silver",
    show_zoombar=True,
    show_lineplot=True,
    show_heatmap=True,
    scale_stat_col=1,
    rename_stat_col=None,
    sites_to_show=None,
):
    """Lineplots and heatmaps of per-site and per-mutation values.

    Parameters
    ----------
    data_df : pandas.DataFrame
        Data to plot. Must have columns "site", "wildtype", "mutant", `stat_col`, and
        `category_col`. The wildtype values (wildtype = mutant) should be included,
        but are not used for the slider filtering or included in site summary lineplot.
    stat_col : str
        Column in `data_df` with statistic to plot.
    category_col : str
        Column in `data_df` with category to facet plots over. You can just create
        a dummy column with some dummy value if you only have one category.
    alphabet : array-like or None
        Alphabet letters in order. If `None`, use natsorted "mutant" col of `data_df`.
    sites : array-like or None
        Sites in order to show. If `None`, use natsorted "site" col of `data_df`.
    addtl_tooltip_stats : None or array-like
        Additional mutation-level stats to show in the heatmap tooltips. Values in
        `addtl_slider_stats` automatically included.
    addtl_slider_stats : None or dict
        Additional stats for which to have a slider, value is initial setting. Ignores
        wildtype and drops it when all mutants have been dropped at site. Null values
        are not filtered.
    addtl_slider_stats_hide_not_filter : None or list
        By default, `addtl_slider_stats` are filtered entirely from data set. If you just
        them excluded from lineplot calculation but marked as filtered on heat map,
        add names of stats to this list.
    init_floor_at_zero : bool
        Initial value for option to put floor of zero on value is `stat_col`.
    init_site_statistic : {'sum', 'mean', 'max', 'min'}
        Initial value for site statistic in lineplot, calculated from `stat_col`.
    cell_size : float
        Size of cells in heatmap
    lineplot_width : float or None
        Overall width of lineplot.
    lineplot_height : float
        Height of line plot.
    site_zoom_bar_width : float
        Width of site zoom bar.
    site_zoom_bar_color_col : float
        Column in `data_df` with which to color zoom bar. Must be the same for all
        entries for a site.
    plot_title : str or None
        Overall plot title.
    show_single_category_label : bool
        Show the category label if just one category.
    category_colors : None or dict
        Map each category to its color, or None to use default. These are the
        colors for **positive** values of `stat_col`.
    heatmap_negative_color : None or str
        Color used for negative values in heatmaps, or None to use default.
    heatmap_color_scheme : None or str
        Heatmap uses this `Vega scheme <https://vega.github.io/vega/docs/schemes>`_
        rather than `category_colors` and `heatmap_negative_color`.
    heatmap_color_scheme_mid_0 : bool
        Set the heatmap color scheme so the domain mid is zero.
    heatmap_max_at_least : None or float
        Make heatmap color max at least this large.
    heatmap_min_at_least : None or float
        Make heatmap color min at least this small, but still set to 0 if floor of zero
        selected.
    heatmap_max_fixed : None or float
        Fix heatmap max to this value, even if it clamps data. Overrides
        `heatmap_max_at_least`.
    heatmap_min_fixed : None or float
        Fix heatmap min to this value, even if it clamps data. Overrides
        `heatmap_min_at_least`.
    site_zoom_bar_color_scheme : str
        If using `site_zoom_bar_color_col`, the
        `Vega color scheme <https://vega.github.io/vega/docs/schemes>`_ to use.
    slider_binding_range_kwargs : dict
        Keyed by keys in ``addtl_slider_stats``, with values being dicts
        giving keyword arguments passed to ``altair.binding_range`` (eg,
        'min', 'max', 'step', etc.
    hide_color : str
        Color given to any cells hidden by `addtl_slider_stats_hide_not_filter`.
    show_zoombar : bool
        Show the zoom bar in the returned chart.
    show_lineplot : bool
        Show the lineplot in the returned chart.
    show_heatmap : bool
        Show the lineplot in the returned chart.
    scale_stat_col : float
        Multiply numbers in `stat_col` by this number before plotting.
    rename_stat_col : None or str
        If a str, rename `stat_col` to this. Also changes y-axis labels.
    sites_to_show : None or dict
        If `None`, all sites are shown. If a dict, can be keyed by "include_range"
        (value a 2-tuple giving first and last site to include, inclusive),
        "include" (list of sites to include), or "exclude" (list of sites to exclude).

    Returns
    -------
    altair.Chart
        Interactive plot.
    """
    if rename_stat_col:
        if rename_stat_col in data_df.columns:
            raise ValueError(f"{rename_stat_col=} already in {data_df.columns=}")
        data_df = data_df.rename(columns={stat_col: rename_stat_col})
        stat_col = rename_stat_col

    basic_req_cols = ["site", "wildtype", "mutant", stat_col, category_col]
    if addtl_tooltip_stats is None:
        addtl_tooltip_stats = []
    if addtl_slider_stats is None:
        addtl_slider_stats = {}
    if addtl_slider_stats_hide_not_filter is None:
        addtl_slider_stats_hide_not_filter = []
    addtl_slider_stats_hide_not_filter = set(addtl_slider_stats).intersection(
        addtl_slider_stats_hide_not_filter
    )
    req_cols = basic_req_cols + addtl_tooltip_stats + list(addtl_slider_stats)
    if site_zoom_bar_color_col:
        req_cols.append(site_zoom_bar_color_col)
    req_cols = list(dict.fromkeys(req_cols))  # https://stackoverflow.com/a/17016257
    if not set(req_cols).issubset(data_df.columns):
        raise ValueError(f"Missing required columns\n{data_df.columns=}\n{req_cols=}")
    if any(c.startswith("_stat") for c in req_cols):  # used for calculated stats
        raise ValueError(f"No columns can start with '_stat' in {data_df.columns=}")
    data_df = (
        data_df[req_cols]
        .reset_index(drop=True)
        .assign(**{stat_col: lambda x: x[stat_col] * scale_stat_col})
    )

    # filter `data_df` by any minimums in `slider_binding_range_kwargs`
    if slider_binding_range_kwargs is None:
        slider_binding_range_kwargs = {}
    for col, col_kwargs in slider_binding_range_kwargs.items():
        if "min" in col_kwargs:
            data_df = data_df[
                (data_df[col] >= col_kwargs["min"])
                | (data_df["wildtype"] == data_df["mutant"])
            ]

    categories = data_df[category_col].unique().tolist()
    show_category_label = show_single_category_label or (len(categories) > 1)

    # set color schemes if use defaults
    if not category_colors:
        if len(categories) > len(DEFAULT_POSITIVE_COLORS):
            raise ValueError("Explicitly set `category_colors` if this many categories")
        category_colors = dict(zip(categories, DEFAULT_POSITIVE_COLORS))
    if not heatmap_negative_color:
        heatmap_negative_color = DEFAULT_NEGATIVE_COLOR

    no_na_cols = basic_req_cols + (
        [site_zoom_bar_color_col] if site_zoom_bar_color_col else []
    )
    if data_df[no_na_cols].isnull().any().any():
        raise ValueError(
            f"`data_df` has NA values in key cols:\n{data_df[no_na_cols].isnull().any()}"
        )

    if alphabet is None:
        alphabet = natsort.natsorted(data_df["mutant"].unique())
    else:
        data_df = data_df.query("mutant in @alphabet")

    if sites is None:
        sites = natsort.natsorted(data_df["site"].unique(), alg=natsort.ns.SIGNED)
    if sites_to_show is not None:
        sites_are_int = data_df["site"].dtype == int
        if "include_range" in sites_to_show:
            start_range, end_range = sites_to_show["include_range"]
            sites = [
                r if sites_are_int else str(r)
                for r in range(start_range, end_range + 1)
            ]
        if "include" in sites_to_show:
            sites = [r if sites_are_int else str(r) for r in sites_to_show["include"]]
        if "exclude" in sites_to_show:
            to_exclude = [
                r if sites_are_int else str(r) for r in sites_to_show["exclude"]
            ]
            sites = [r for r in sites if r not in to_exclude]
    data_df = data_df.query("site in @sites")
    sites = [site for site in sites if site in set(data_df["site"])]

    # order sites:
    # https://github.com/dms-vep/dms-vep-pipeline/issues/53#issuecomment-1227817963
    site_to_i = {site: i for i, site in enumerate(sites)}
    data_df = data_df.assign(_stat_site_order=lambda x: x["site"].map(site_to_i))

    # get tooltips for heatmap
    float_cols = [c for c in req_cols if data_df[c].dtype == float]
    heatmap_tooltips = [
        alt.Tooltip(c, type="quantitative", format=".3g")
        if c in float_cols
        else alt.Tooltip(c, type="nominal")
        for c in req_cols
        if c != category_col or show_category_label
    ]

    # make floor at zero selection, setting floor to either 0 or min in data (no floor)
    min_stat = data_df[stat_col].min()  # used as min in heatmap when not flooring at 0
    if heatmap_min_at_least is not None:
        min_stat = min(min_stat, heatmap_min_at_least)
    if heatmap_min_fixed is not None:
        min_stat = heatmap_min_fixed
    max_stat = data_df[stat_col].max()  # used as max in heatmap
    if heatmap_max_at_least is not None:
        max_stat = max(max_stat, heatmap_max_at_least)
    if heatmap_max_fixed is not None:
        max_stat = heatmap_max_fixed
    floor_at_zero = alt.selection_point(
        name="floor_at_zero",
        bind=alt.binding_radio(
            options=[0, min_stat],
            labels=["yes", "no"],
            name=f"floor {stat_col} at zero",
        ),
        fields=["floor"],
        value=[{"floor": 0 if init_floor_at_zero else min_stat}],
    )

    # create sliders for max of statistic at site and any additional sliders
    sliders = {}
    for slider_stat, init_slider_stat in addtl_slider_stats.items():
        binding_range_kwargs = {
            "min": data_df[slider_stat].min(),
            "max": data_df[slider_stat].max(),
            "name": f"minimum {slider_stat}",
        }
        if slider_stat in slider_binding_range_kwargs:
            binding_range_kwargs.update(slider_binding_range_kwargs[slider_stat])
        sliders[slider_stat] = alt.selection_point(
            fields=["cutoff"],
            value=[{"cutoff": init_slider_stat}],
            bind=alt.binding_range(**binding_range_kwargs),
        )
    sliders["_stat_site_max"] = alt.selection_point(
        fields=["cutoff"],
        value=[{"cutoff": min_stat}],
        bind=alt.binding_range(
            name=f"minimum max of {stat_col} at site",
            min=min_stat,
            max=max_stat,
        ),
    )

    # whether to show line on line plot
    line_selection = alt.selection_point(
        bind=alt.binding_radio(
            options=[True, False],
            labels=["yes", "no"],
            name="show line on site plot",
        ),
        fields=["_stat_show_line"],
        value=[{"_stat_show_line": True}],
    )

    # create site zoom bar
    site_brush = alt.selection_interval(
        encodings=["x"],
        mark=alt.BrushConfig(stroke="black", strokeWidth=2),
    )
    if site_zoom_bar_color_col:
        site_zoom_bar_df = data_df[
            ["site", "_stat_site_order", site_zoom_bar_color_col]
        ].drop_duplicates()
        if any(site_zoom_bar_df.groupby("site").size() > 1):
            raise ValueError(f"multiple {site_zoom_bar_color_col=} values for sites")
    else:
        site_zoom_bar_df = data_df[["site", "_stat_site_order"]].drop_duplicates()
    site_zoom_bar = (
        alt.Chart(site_zoom_bar_df)
        .mark_rect()
        .encode(
            x=alt.X(
                "site:O",
                sort=alt.EncodingSortField(field="_stat_site_order", order="ascending"),
            ),
            color=(
                alt.Color(
                    site_zoom_bar_color_col,
                    type="nominal",
                    scale=alt.Scale(scheme=site_zoom_bar_color_scheme),
                    legend=alt.Legend(orient="left"),
                    sort=(
                        site_zoom_bar_df.set_index("site")
                        .loc[sites][site_zoom_bar_color_col]
                        .unique()
                    ),
                )
                if site_zoom_bar_color_col
                else alt.value("gray")
            ),
            tooltip=[c for c in site_zoom_bar_df.columns if not c.startswith("_stat")],
        )
        .mark_rect()
        .add_params(site_brush)
        .properties(width=site_zoom_bar_width, height=cell_size, title="site zoom bar")
    )

    # to make data in Chart smaller, access properties that are same across all sites
    # or categories via a transform_lookup. Make data frames with columns to do that.
    lookup_dfs = {}
    for lookup_col in ["site", category_col]:
        cols_to_lookup = [
            c
            for c in data_df.columns
            if all(data_df.groupby(lookup_col)[c].nunique(dropna=False) == 1)
            if c not in ["site", category_col]
        ]
        if cols_to_lookup:
            lookup_dfs[lookup_col] = data_df[
                [lookup_col, *cols_to_lookup]
            ].drop_duplicates()
            assert len(lookup_dfs[lookup_col]) == data_df[lookup_col].nunique()
            data_df = data_df.drop(columns=cols_to_lookup)

    # make the base chart that holds the data and common elements
    base_chart = alt.Chart(data_df)
    for lookup_col, lookup_df in lookup_dfs.items():
        base_chart = base_chart.transform_lookup(
            lookup=lookup_col,
            from_=alt.LookupData(
                data=lookup_df,
                key=lookup_col,
                fields=[c for c in lookup_df.columns if c != lookup_col],
            ),
        )
    # convert null values to NaN so they show as NaN in tooltips rather than as 0.0
    for col in float_cols:
        base_chart = base_chart.transform_calculate(
            **{
                col: alt.expr.if_(
                    alt.expr.isFinite(alt.datum[col]),
                    alt.datum[col],
                    alt.expr.NaN,
                )
            }
        )

    # Transforms on base chart. The "_stat" columns is floor transformed stat_col.
    base_chart = base_chart.transform_calculate(
        _stat=alt.expr.max(alt.datum[stat_col], floor_at_zero["floor"]),
    )

    # get stats to hide, not filter
    if addtl_slider_stats_hide_not_filter:
        # https://stackoverflow.com/a/61502057/4191652
        sel = [
            alt.datum[slider_stat]
            <= (sliders[slider_stat]["cutoff"] - 1e-6)  # roundtol
            for slider_stat in addtl_slider_stats_hide_not_filter
        ]
        base_chart = base_chart.transform_calculate(
            _stat_hide=functools.reduce(operator.or_, sel)
        )
    else:
        base_chart = base_chart.transform_calculate(_stat_hide="false")
    # Filter data using slider stat
    assert list(sliders)[-1] == "_stat_site_max"  # last for right operation order
    for slider_stat, slider in sliders.items():
        if slider_stat == "_stat_site_max":
            base_chart = base_chart.transform_calculate(
                _stat_not_hidden=alt.expr.if_(
                    alt.datum["_stat_hide"],
                    data_df[stat_col].min(),
                    alt.datum["_stat"],
                ),
            ).transform_joinaggregate(
                _stat_site_max="max(_stat_not_hidden)",
                groupby=["site"],
            )
        if slider_stat not in addtl_slider_stats_hide_not_filter:
            base_chart = base_chart.transform_filter(
                (alt.datum[slider_stat] >= (slider["cutoff"] - 1e-6))  # rounding tol
                | ~alt.expr.isFinite(
                    alt.datum[slider_stat]
                )  # do not filter null values
            )
    # Remove any sites that are only wildtype and filter with site zoom brush
    base_chart = (
        base_chart.transform_calculate(
            _stat_not_wildtype=alt.datum.wildtype != alt.datum.mutant
        )
        .transform_joinaggregate(
            _stat_site_has_non_wildtype="max(_stat_not_wildtype)",
            groupby=["site"],
        )
        .transform_filter(alt.datum["_stat_site_has_non_wildtype"])
        .transform_filter(site_brush)
    )

    # make the site chart
    site_statistics = ["sum", "mean", "max", "min"]
    if init_site_statistic not in site_statistics:
        raise ValueError(f"invalid {init_site_statistic=}")
    if set(site_statistics).intersection(req_cols):
        raise ValueError(f"`data_df` cannot have these columns:\n{site_statistics}")
    site_stat = alt.selection_point(
        bind=alt.binding_radio(
            labels=site_statistics,
            options=[f"_stat_{stat}" for stat in site_statistics],
            name=f"site {stat_col} statistic",
        ),
        fields=["_stat_site_stat"],
        value=[{"_stat_site_stat": f"_stat_{init_site_statistic}"}],
        name="site_stat",
    )
    site_prop_cols = lookup_dfs["site"].columns if "site" in lookup_dfs else ["site"]
    lineplot_base = (
        base_chart.transform_filter(
            (alt.datum.wildtype != alt.datum.mutant) & ~alt.datum["_stat_hide"]
        )
        .transform_aggregate(
            **{f"_stat_{stat}": f"{stat}(_stat)" for stat in site_statistics},
            groupby=[*site_prop_cols, category_col],
        )
        .transform_fold(
            [f"_stat_{stat}" for stat in site_statistics],
            ["_stat_site_stat", "_stat_site_val"],
        )
        .transform_filter(site_stat)
        .encode(
            x=alt.X(
                "site:O",
                sort=alt.EncodingSortField(field="_stat_site_order", order="ascending"),
            ),
            y=alt.Y(
                "_stat_site_val:Q",
                scale=alt.Scale(zero=True),
                title=f"site {stat_col}",
            ),
            color=alt.Color(
                category_col,
                scale=alt.Scale(
                    domain=categories,
                    range=[category_colors[c] for c in categories],
                ),
                legend=alt.Legend(orient="left") if show_category_label else None,
            ),
            tooltip=[
                "site",
                *([category_col] if show_category_label else []),
                alt.Tooltip("_stat_site_val:Q", format=".3g", title=f"site {stat_col}"),
                *[
                    f"{c}:N"
                    for c in site_prop_cols
                    if c != "site" and not c.startswith("_stat")
                ],
            ],
        )
    )
    site_lineplot = (
        (
            (
                lineplot_base.mark_line(size=1, opacity=0.7)
                .transform_calculate(_stat_show_line="true")
                .transform_filter(line_selection)
            )
            + lineplot_base.mark_circle(opacity=0.7)
        )
        .add_params(site_stat, line_selection)
        .properties(width=lineplot_width, height=lineplot_height)
    )

    # make base chart for heatmaps
    heatmap_base = base_chart.encode(
        y=alt.Y(
            "mutant",
            sort=alphabet,
            scale=alt.Scale(domain=alphabet),
            title=None,
        ),
    )

    # wildtype text marks for heatmap
    heatmap_wildtype = (
        heatmap_base.encode(
            x=alt.X(
                "site:O",
                sort=alt.EncodingSortField(field="_stat_site_order", order="ascending"),
            ),
        )
        .transform_filter(alt.datum.wildtype == alt.datum.mutant)
        .mark_text(text="x", color="black")
    )

    # background fill for missing values in heatmap, imputing dummy stat
    # to get all cells
    heatmap_bg = (
        heatmap_base.encode(
            x=alt.X(
                "site:O",
                sort=alt.EncodingSortField(field="_stat_site_order", order="ascending"),
            )
        )
        .transform_impute(
            impute="_stat_dummy",
            key="mutant",
            keyvals=alphabet,
            groupby=["site"],
            value=None,
        )
        .mark_rect(color="#E0E0E0")
    )

    # Make heatmaps for each category and vertically concatenate. We do this in loop
    # rather than faceting to enable compound chart w wildtype marks and category
    # specific coloring.
    heatmaps = []
    for category in categories:
        heatmap_no_color = (
            heatmap_base.transform_filter(alt.datum[category_col] == category)
            .encode(
                x=alt.X(
                    "site:O",
                    sort=alt.EncodingSortField(
                        field="_stat_site_order",
                        order="ascending",
                    ),
                    # only show ticks and axis title on bottom most category
                    axis=alt.Axis(
                        labels=category == categories[-1],
                        ticks=category == categories[-1],
                        title="site" if category == categories[-1] else None,
                    ),
                ),
                stroke=alt.value("black"),
                tooltip=heatmap_tooltips,
            )
            .mark_rect()
            .properties(
                width=alt.Step(cell_size),
                height=alt.Step(cell_size),
                title=alt.TitleParams(
                    category if show_category_label else "",
                    color=category_colors[category],
                    anchor="middle",
                    orient="left",
                ),
            )
        )
        heatmap = heatmap_no_color.transform_filter(~alt.datum["_stat_hide"]).encode(
            color=alt.Color(
                "_stat:Q",
                legend=alt.Legend(
                    orient="left",
                    title=stat_col,
                    titleOrient="left",
                    gradientLength=100,
                    gradientStrokeColor="black",
                    gradientStrokeWidth=0.5,
                ),
                scale=alt.Scale(
                    domainMax=max_stat,
                    domainMin=alt.ExprRef("floor_at_zero.floor"),
                    zero=True,
                    nice=False,
                    clamp=True,
                    type="linear",
                    **({"domainMid": 0} if heatmap_color_scheme_mid_0 else {}),
                    **(
                        {"scheme": heatmap_color_scheme}
                        if heatmap_color_scheme
                        else {
                            "range": (
                                color_gradient_hex(
                                    heatmap_negative_color, "white", n=20
                                )
                                + color_gradient_hex(
                                    "white", category_colors[category], n=20
                                )[1:]
                            )
                        }
                    ),
                ),
            ),
        )
        heatmap_hide = heatmap_no_color.transform_filter(
            alt.datum["_stat_hide"]
        ).encode(color=alt.value(hide_color))
        heatmaps.append(heatmap_bg + heatmap_hide + heatmap + heatmap_wildtype)
    heatmaps = alt.vconcat(*heatmaps, spacing=10).resolve_scale(
        x="shared",
        color="shared"
        if heatmap_color_scheme or len(categories) == 1
        else "independent",
    )

    chartlist = []
    if show_zoombar:
        chartlist.append(site_zoom_bar)
    if show_lineplot:
        chartlist.append(site_lineplot)
    if show_heatmap:
        chartlist.append(heatmaps)
    chart = (
        alt.vconcat(*chartlist)
        .add_params(floor_at_zero, site_brush, *sliders.values())
        .configure(padding=10)
        .configure_axis(labelOverlap="parity", grid=False)
        .resolve_scale(color="independent")
    )

    if plot_title:
        chart = chart.properties(
            title=alt.TitleParams(
                plot_title,
                anchor="start",
                align="left",
                fontSize=16,
            ),
        )

    return chart


if __name__ == "__main__":
    import doctest

    doctest.testmod()
