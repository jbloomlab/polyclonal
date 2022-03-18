"""
==========
plot
==========

Plotting functions.

"""


import itertools

import altair as alt

import matplotlib.colors

import numpy

import pandas as pd


alt.data_transformers.disable_max_rows()


TAB10_COLORS_NOGRAY = tuple(
    c for c in matplotlib.colors.TABLEAU_COLORS.values() if c != "#7f7f7f"
)
"""tuple: Tableau 10 color palette without gray."""


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
    stat : str or array-like
        Statistic in `activity_wt_df` to plot as activity, or list of dropdown options.
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

    if isinstance(stat, str):
        if stat not in activity_wt_df.columns:
            raise ValueError(f"{stat=} not in {activity_wt_df.columns=}")
        df = activity_wt_df
        stat_selection = None
    elif len(stat) > 0:
        if not set(stat).issubset(activity_wt_df.columns):
            raise ValueError(f"{stat=} not all in {activity_wt_df.columns=}")
        df = activity_wt_df.melt(
            id_vars=["epitope"],
            value_vars=stat,
            var_name="statistic",
            value_name="activity",
        )
        stat_selection = alt.selection_single(
            fields=["statistic"],
            init={"statistic": stat[0]},
            bind=alt.binding_select(options=stat, name="statistic"),
        )
        stat = "activity"
    else:
        raise ValueError(f"invalid {stat=}")

    if error_stat is not None:
        if error_stat not in activity_wt_df.columns:
            raise ValueError(f"{error_stat=} not in {activity_wt_df.columns=}")
        assert not {"_upper", "_lower"}.intersection(activity_wt_df.columns)
        df = df.merge(activity_wt_df[["epitope", error_stat]]).assign(
            _lower=lambda x: x[stat] - x[error_stat],
            _upper=lambda x: x[stat] + x[error_stat],
        )

    baseplot = (
        alt.Chart(df)
        .encode(
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
        .properties(width=width, height={"step": height_per_bar})
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

    if stat_selection is not None:
        barplot = barplot.add_selection(stat_selection).transform_filter(stat_selection)

    return barplot.configure_axis(grid=False)


def mut_escape_lineplot(
    *,
    mut_escape_site_summary_df,
    bootstrapped_data=False,
    epitope_colors,
    epitopes=None,
    all_sites=True,
    share_ylims=True,
    height=100,
    width=900,
    init_metric="mean",
    zoom_bar_width=500,
):
    r"""Line plots of mutation escape :math:`\beta_{m,e}` at each site.

    Parameters
    -----------
    mut_escape_site_summary_df : pandas.DataFrame
        Site-level escape in format of
        :attr:`polyclonal.polyclonal.Polyclonal.mut_escape_site_summary_df`.
    bootstrapped_data : bool
        `mut_escape_site_summary_df` is from a bootstrapped model,
        :class:`polyclonal.polyclonal_collection.PolyclonalCollection`.
    epitope_colors : dict
        Maps each epitope name to its color.
    epitopes : array-like or None
        Make plots for these epitopes. If `None`, use all epitopes.
    all_sites : bool
        Plot all sites in range from first to last site even if some
        have no data.
    share_ylims : bool
        Should plots for all epitopes share same y-limits?
    height : float
        Height per facet.
    width : float
        Width of plot.
    init_metric : str
        Metric to show initially (others can be selected by dropdown). One of
        metrics in :attr:`polyclonal.polyclonal.Polyclonal.mut_escape_site_summary_df`.
    zoom_bar_width : float
        Width of zoom bar

    Returns
    -------
    altair.Chart
        Interactive plot.

    """
    if epitopes is None:
        epitopes = mut_escape_site_summary_df["epitope"].unique().tolist()
    elif not set(epitopes).issubset(mut_escape_site_summary_df["epitope"]):
        raise ValueError("invalid entries in `epitopes`")

    df = mut_escape_site_summary_df.query("epitope in @epitopes")
    if bootstrapped_data:
        escape_metrics = df["metric"].unique().tolist()
    else:
        escape_metrics = [
            m for m in df.columns if m not in {"epitope", "site", "wildtype"}
        ]

    sites = df["site"].unique().tolist()
    if all_sites:
        sites = list(range(min(sites), max(sites) + 1))

    # fill any missing sites
    df = df.merge(
        pd.DataFrame(itertools.product(sites, epitopes), columns=["site", "epitope"]),
        on=["site", "epitope"],
        how="right",
    )

    if bootstrapped_data:
        df = df.melt(
            id_vars=[
                "epitope",
                "site",
                "metric",
                "std",
                "frac_bootstrap_replicates",
            ],
            value_vars=["mean", "median"],
            var_name="statistic",
            value_name="escape",
        )
        statistic_selection = alt.selection_single(
            fields=["statistic"],
            bind=alt.binding_select(options=df["statistic"].unique()),
            name="bootstrap",
            init={"statistic": df["statistic"].unique()[0]},
        )
        index = ["site", "metric", "statistic"]
    else:
        df = df.melt(
            id_vars=["epitope", "site"],
            value_vars=escape_metrics,
            var_name="metric",
            value_name="escape",
        )
        statistic_selection = None
        index = ["site", "metric"]

    df = (
        df.pivot_table(index=index, values="escape", columns="epitope", dropna=False)
        .reset_index()
    )

    metric_selection = alt.selection_single(
        fields=["metric"],
        bind=alt.binding_select(options=escape_metrics),
        name="escape",
        init={"metric": init_metric},
    )

    zoom_brush = alt.selection_interval(
        encodings=["x"],
        mark=alt.BrushConfig(stroke="black", strokeWidth=2),
    )
    zoom_bar = (
        alt.Chart(df)
        .mark_rect(color="gray")
        .encode(x="site:O")
        .add_selection(zoom_brush)
        .properties(
            width=zoom_bar_width,
            height=15,
            title="site zoom bar",
        )
    )

    site_selector = alt.selection(
        type="single", on="mouseover", fields=["site"], empty="none"
    )

    # add wildtypes and potential frac_bootstrap_replicates
    cols = ["site", "wildtype"]
    if bootstrapped_data:
        cols.append("frac_bootstrap_replicates")
    df = df.merge(
        mut_escape_site_summary_df[cols].drop_duplicates(),
        how="left",
        on="site",
        validate="many_to_one",
    )

    charts = []
    for epitope in epitopes:
        base = alt.Chart(df).encode(
            x=alt.X(
                "site:O",
                title=("site" if epitope == epitopes[-1] else None),
                axis=(alt.Axis() if epitope == epitopes[-1] else None),
            ),
            y=alt.Y(
                epitope,
                type="quantitative",
                title="escape",
                scale=alt.Scale(),
            ),
            tooltip=[
                alt.Tooltip("site:O"),
                alt.Tooltip("wildtype:N"),
                *[alt.Tooltip(f"{epitope}:Q", format=".2f") for epitope in epitopes],
                *([alt.Tooltip("frac_bootstrap_replicates", format=".2f")]
                   if bootstrapped_data else []
                ),
            ],
        )
        # in case some sites missing values, background thin transparent
        # over which we put darker foreground for measured points
        background = base.transform_filter(f"isValid(datum['{epitope}'])").mark_line(
            opacity=0.5, size=1, color=epitope_colors[epitope]
        )
        foreground = base.mark_line(opacity=1, size=1.5, color=epitope_colors[epitope])
        foreground_circles = (
            base.mark_circle(opacity=1, color=epitope_colors[epitope])
            .encode(
                size=alt.condition(site_selector, alt.value(75), alt.value(25)),
                stroke=alt.condition(
                    site_selector, alt.value("black"), alt.value(None)
                ),
            )
            .add_selection(site_selector)
        )
        charts.append(
            (background + foreground + foreground_circles)
            .add_selection(metric_selection)
            .transform_filter(metric_selection)
            .transform_filter(zoom_brush)
            .properties(
                title=alt.TitleParams(
                    f"{epitope} epitope", color=epitope_colors[epitope]
                ),
                width=width,
                height=height,
            )
        )
        if statistic_selection is not None:
            charts[-1] = (
                charts[-1].add_selection(statistic_selection)
                .transform_filter(statistic_selection)
            )

    return (
        alt.vconcat(
            zoom_bar,
            (
                alt.vconcat(*charts, spacing=10).resolve_scale(
                    y="shared" if share_ylims else "independent"
                )
            ),
            spacing=10,
        )
        .configure(padding={"left": 15, "top": 5, "right": 5, "bottom": 5})
        .configure_axis(grid=False, labelOverlap="parity")
        .configure_title(anchor="start", fontSize=14)
    )


def mut_escape_heatmap(
    *,
    mut_escape_df,
    alphabet,
    epitope_colors,
    epitopes=None,
    stat="escape",
    error_stat=None,
    addtl_tooltip_stats=None,
    all_sites=True,
    all_alphabet=True,
    floor_color_at_zero=True,
    share_heatmap_lims=True,
    cell_size=13,
    zoom_bar_width=500,
):
    r"""Heatmaps of the mutation escape values, :math:`\beta_{m,e}`.

    Parameters
    ----------
    mut_escape_df : pandas.DataFrame
        Mutation-level escape in format of
        :attr:`polyclonal.polyclonal.Polyclonal.mut_escape_df`.
    alphabet : array-like or None
        Alphabet letters (e.g., amino acids) in order to plot them.
    epitope_colors : dict
        Maps each epitope name to its color.
    epitopes : array-like or None
        Make plots for these epitopes. If `None`, use all epitopes.
    stat : str or array-like
        Statistic in `mut_escape_df` to plot as escape, or list of dropdown options.
    error_stat : str or None
        Measure of error to display in tooltip.
    addtl_tooltip_stats : list or None
        Additional mutation-level stats to show in tooltip.
    all_sites : bool
        Plot all sites in range from first to last site even if some
        have no data.
    all_alphabet : bool
        Plot all letters in the alphabet (e.g., amino acids) even if some
        have no data.
    floor_color_at_zero : bool
        Set lower limit to color scale as zero, even if there are negative
        values or if minimum is >0.
    share_heatmap_lims : bool
        If `True`, let all epitopes share the same limits in color scale.
        If `False`, scale each epitopes colors to the min and max stat
        values for that epitope.
    cell_size : float
        Size of cells in heatmap.
    zoom_bar_width : float
        Width of zoom bar

    Returns
    -------
    altair.Chart
        Interactive heat maps.

    """
    if epitopes is None:
        epitopes = mut_escape_df["epitope"].unique().tolist()
    elif not set(epitopes).issubset(mut_escape_df["epitope"]):
        raise ValueError("invalid entries in `epitopes`")

    df = mut_escape_df.query("epitope in @epitopes")

    # get alphabet and sites, expanding to all if needed
    extrachars = set(df["mutant"]).union(set(df["wildtype"])) - set(alphabet)
    if extrachars:
        raise ValueError(
            "`mut_escape_df` has letters not in `alphabet`:\n" + str(extrachars)
        )
    if not all_alphabet:
        alphabet = [c for c in alphabet if c in set(df["mutant"]) + set(df["wildtype"])]
    sites = df["site"].unique().tolist()
    if all_sites:
        sites = list(range(min(sites), max(sites) + 1))

    wts = mut_escape_df.set_index("site")["wildtype"].to_dict()

    if isinstance(stat, str):
        if stat not in df:
            raise ValueError(f"{stat=} not in {df.columns=}")
        index = ["site", "mutant"]  # for later pivoting
        product_cols = [sites, alphabet]  # for later filling gaps
        stat_selection = None
    elif len(stat) > 0:
        if not set(stat).issubset(df.columns):
            raise ValueError(f"{stat=} not in {df.columns=}")
        df = df.melt(
            id_vars=[c for c in df.columns if c not in stat],
            value_vars=stat,
            var_name="statistic",
            value_name="escape",
        )
        index = ["site", "mutant", "statistic"]  # for later pivoting
        product_cols = [sites, alphabet, stat]  # for later filling gaps
        stat_selection = alt.selection_single(
            fields=["statistic"],
            init={"statistic": stat[0]},
            bind=alt.binding_select(options=stat, name="statistic"),
        )
        stat = "escape"
    else:
        raise ValueError(f"invalid {stat=}")

    # get labels for escape to show on tooltips, potentially with error
    if error_stat is not None:
        if error_stat not in df.columns:
            raise ValueError(f"{error_stat=} not in {df.columns=}")
        label_df = df.assign(
            label=lambda x: x.apply(
                lambda r: f"{r[stat]:.2f} +/- {r[error_stat]:.2f}",
                axis=1,
            )
        )
    else:
        label_df = df.assign(label=lambda x: x[stat].map(lambda s: f"{s:.2f}"))
    label_df = label_df.pivot_table(
        index=index,
        values="label",
        columns="epitope",
        aggfunc=lambda x: " ".join(x),
    ).rename(columns={e: f"{e} epitope" for e in epitopes})

    df = (
        df.pivot_table(index=index, values=stat, columns="epitope")
        .reset_index()
        .merge(
            pd.DataFrame(
                itertools.product(*product_cols),
                columns=index,
            ),
            how="right",
        )
        .assign(
            wildtype=lambda x: x["site"].map(wts),
            mutation=lambda x: (
                x["wildtype"].fillna("") + x["site"].astype(str) + x["mutant"]
            ),
            # mark wildtype cells with a `x`
            wildtype_char=lambda x: (x["mutant"] == x["wildtype"]).map(
                {True: "x", False: ""}
            ),
        )
        .merge(label_df, how="left", on=index, validate="one_to_one")
    )
    # wildtype has escape of 0 by definition
    for epitope in epitopes:
        df[epitope] = df[epitope].where(df["mutant"] != df["wildtype"], 0)
        label_col = f"{epitope} epitope"
        df[label_col] = df[label_col].where(df["mutant"] != df["wildtype"], "0")

    # zoom bar to put at top
    zoom_brush = alt.selection_interval(
        encodings=["x"], mark=alt.BrushConfig(stroke="black", strokeWidth=2)
    )
    zoom_bar = (
        alt.Chart(df)
        .mark_rect(color="gray")
        .encode(x="site:O")
        .add_selection(zoom_brush)
        .properties(
            width=zoom_bar_width,
            height=15,
            title="site zoom bar",
        )
    )

    add_tooltips = []
    if addtl_tooltip_stats is not None:
        if not set(addtl_tooltip_stats).issubset(mut_escape_df.columns):
            raise ValueError(f"{addtl_tooltip_stats=} not in {mut_escape_df.columns=}")
        df = df.merge(
            mut_escape_df[["site", "mutant"] + addtl_tooltip_stats].drop_duplicates(),
            on=["site", "mutant"],
            how="left",
            validate="many_to_one",
        )
        add_tooltips = [
            alt.Tooltip(c, format=".2g") if mut_escape_df[c].dtype == float else c
            for c in addtl_tooltip_stats
        ]

    # select cells
    cell_selector = alt.selection_single(on="mouseover", empty="none")

    # make list of heatmaps for each epitope
    charts = [zoom_bar]
    # base chart
    base = alt.Chart(df).encode(
        x=alt.X("site:O"),
        y=alt.Y("mutant:O", sort=alphabet),
    )
    for epitope in epitopes:
        # heatmap for cells with data
        if share_heatmap_lims:
            vals = df[list(epitopes)].values
        else:
            vals = df[epitope].values
        escape_max = numpy.nanmax(vals)
        if floor_color_at_zero:
            escape_min = 0
        else:
            escape_min = numpy.nanmin(vals)
        if not (escape_min < escape_max):
            raise ValueError("escape min / max do not span a valid range")
        heatmap = base.mark_rect().encode(
            color=alt.Color(
                epitope,
                type="quantitative",
                scale=alt.Scale(
                    range=color_gradient_hex("white", epitope_colors[epitope], 10),
                    type="linear",
                    domain=(escape_min, escape_max),
                    clamp=True,
                ),
                legend=alt.Legend(
                    orient="left",
                    title="gray is n.d.",
                    titleFontWeight="normal",
                    gradientLength=100,
                    gradientStrokeColor="black",
                    gradientStrokeWidth=0.5,
                ),
            ),
            stroke=alt.value("black"),
            strokeWidth=alt.condition(cell_selector, alt.value(2.5), alt.value(0.2)),
            tooltip=["mutation"] + [f"{e} epitope:N" for e in epitopes] + add_tooltips,
        )
        # nulls for cells with missing data
        nulls = (
            base.mark_rect()
            .transform_filter(f"!isValid(datum['{epitope}'])")
            .mark_rect(opacity=0.25)
            .encode(
                alt.Color(f"{stat}:N", scale=alt.Scale(scheme="greys"), legend=None),
            )
        )
        # mark wildtype cells
        wildtype = base.mark_text(color="black").encode(
            text=alt.Text("wildtype_char:N")
        )
        # combine the elements
        charts.append(
            (heatmap + nulls + wildtype)
            .interactive()
            .add_selection(cell_selector)
            .transform_filter(zoom_brush)
            .properties(
                title=alt.TitleParams(
                    f"{epitope} epitope", color=epitope_colors[epitope]
                ),
                width={"step": cell_size},
                height={"step": cell_size},
            )
        )
        if stat_selection is not None:
            charts[-1] = (
                charts[-1]
                .add_selection(stat_selection)
                .transform_filter(stat_selection)
            )

    chart = (
        alt.vconcat(
            *charts,
            spacing=0,
        )
        .configure_axis(labelOverlap="parity")
        .configure_title(anchor="start", fontSize=14)
    )

    return chart


if __name__ == "__main__":
    import doctest

    doctest.testmod()
