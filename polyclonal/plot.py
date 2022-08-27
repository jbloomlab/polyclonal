"""
==========
plot
==========

Plotting functions.

"""


import altair as alt

import matplotlib.colors

import natsort


alt.data_transformers.disable_max_rows()


TAB10_COLORS_NOGRAY = tuple(
    c for c in matplotlib.colors.TABLEAU_COLORS.values() if c != "#7f7f7f"
)
"""tuple: Tableau 10 color palette without gray."""

DEFAULT_POSITIVE_COLORS = ("#0072B2", "#009E73", "#CC79A7", "#56B4E9", "#F0E442")
"""tuple: Some cbPalette colors in hex: dark blue, green, purple, light blue,yellow."""

DEFAULT_NEGATIVE_COLOR = "#E69F00"
"""str: Orange from cbPalette color in hex."""


def color_gradient_hex(start, end, n):
    """Get a list of colors linearly spanning a range.

    Parameters
    -----------
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
    site_zoom_bar_color_scheme="set3",
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
        Sites in order. If `None`, use natsorted "site" col of `data_df`.
    addtl_tooltip_stats : None or array-like
        Additional mutation-level stats to show in the heatmap tooltips. Values in
        `addtl_slider_stats` automatically included.
    addtl_slider_stats : None or dict
        Additional stats for which to have a slider, value is initial setting. Ignores
        wildtype and drops it when all mutants have been dropped at site. Null values
        are not filtered.
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
    site_zoom_bar_color_scheme : str
        If using `site_zoom_bar_color_col`, the
        `Vega color scheme <https://vega.github.io/vega/docs/schemes>`_ to use.
    """
    basic_req_cols = ["site", "wildtype", "mutant", stat_col, category_col]
    if addtl_tooltip_stats is None:
        addtl_tooltip_stats = []
    if addtl_slider_stats is None:
        addtl_slider_stats = {}
    req_cols = basic_req_cols + addtl_tooltip_stats + list(addtl_slider_stats)
    if site_zoom_bar_color_col:
        req_cols.append(site_zoom_bar_color_col)
    req_cols = list(dict.fromkeys(req_cols))  # https://stackoverflow.com/a/17016257
    if not set(req_cols).issubset(data_df.columns):
        raise ValueError(f"Missing required columns\n{data_df.columns=}\n{req_cols=}")
    if any(c.startswith("_stat") for c in req_cols):  # used for calculated stats
        raise ValueError(f"No columns can start with '_stat' in {data_df.columns=}")
    data_df = data_df[req_cols].reset_index(drop=True)

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
    else:
        data_df = data_df.query("site in @sites")
    # order sites:
    # https://github.com/dms-vep/dms-vep-pipeline/issues/53#issuecomment-1227817963
    data_df["_stat_site_order"] = data_df["site"].map(
        {site: i for i, site in enumerate(sites)}
    )

    # get tooltips for heatmap
    heatmap_tooltips = [
        alt.Tooltip(c, type="quantitative", format=".3g")
        if data_df[c].dtype == float
        else alt.Tooltip(c, type="nominal")
        for c in req_cols
        if c != category_col or show_category_label
    ]

    # make floor at zero selection, setting floor to either 0 or min in data (no floor)
    min_stat = data_df[stat_col].min()  # used as min in heatmap when not flooring at 0
    if heatmap_min_at_least is not None:
        min_stat = min(min_stat, heatmap_min_at_least)
    max_stat = data_df[stat_col].max()  # used as max in heatmap
    if heatmap_max_at_least is not None:
        max_stat = max(max_stat, heatmap_max_at_least)
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

    # create sliders for site statistic value and any additional sliders
    site_statistics = ["sum", "mean", "max", "min"]
    sliders = {
        "_stat_site_max": alt.selection_point(
            fields=["cutoff"],
            value=[{"cutoff": min_stat}],
            bind=alt.binding_range(
                name=f"minimum max of {stat_col} at site",
                min=min_stat,
                max=max_stat,
            ),
        ),
    }
    for slider_stat, init_slider_stat in addtl_slider_stats.items():
        sliders[slider_stat] = alt.selection_point(
            fields=["cutoff"],
            value=[{"cutoff": init_slider_stat}],
            bind=alt.binding_range(
                min=data_df[slider_stat].min(),
                max=data_df[slider_stat].max(),
                name=f"minimum {slider_stat}",
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
        site_zoom_bar_df = data_df[["site", site_zoom_bar_color_col]].drop_duplicates()
        if any(site_zoom_bar_df.groupby("site").size() > 1):
            raise ValueError(f"multiple {site_zoom_bar_color_col=} values for sites")
    else:
        site_zoom_bar_df = data_df[["site"]].drop_duplicates()
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
            tooltip=[
                c
                for c in site_zoom_bar_df.columns
                if c != category_col or show_category_label
            ],
        )
        .mark_rect()
        .add_parameter(site_brush)
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

    # Transforms on base chart. The "_stat" columns is floor transformed stat_col.
    base_chart = base_chart.transform_calculate(
        _stat=alt.expr.max(alt.datum[stat_col], floor_at_zero["floor"]),
    ).transform_joinaggregate(_stat_site_max="max(_stat)", groupby=["site"])

    # Filter data using slider stat
    for slider_stat, slider in sliders.items():
        base_chart = base_chart.transform_filter(
            (alt.datum[slider_stat] >= slider["cutoff"] - 1e-6)  # add rounding tol
            | ~alt.expr.isNumber(alt.datum[slider_stat])  # do not filter null values
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
        base_chart.transform_filter(alt.datum.wildtype != alt.datum.mutant)
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
                *[f"{c}:N" for c in site_prop_cols if c != "site"],
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
        .add_parameter(site_stat, line_selection)
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
        .mark_rect(color="gray", opacity=0.25)
    )

    # Make heatmaps for each category and vertically concatenate. We do this in loop
    # rather than faceting to enable compound chart w wildtype marks and category
    # specific coloring.
    heatmaps = alt.vconcat(
        *[
            heatmap_bg
            + heatmap_base.transform_filter(alt.datum[category_col] == category)
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
            + heatmap_wildtype
            for category in categories
        ],
        spacing=10,
    ).resolve_scale(
        x="shared",
        color="shared"
        if heatmap_color_scheme or len(categories) == 1
        else "independent",
    )

    chart = (
        alt.vconcat(site_zoom_bar, site_lineplot, heatmaps)
        .add_parameter(floor_at_zero, site_brush, *sliders.values())
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
