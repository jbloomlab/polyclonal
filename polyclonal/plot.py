"""
==========
plot
==========

Plotting functions.

"""


import itertools
import math

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


def mut_escape_lineplot(
    *,
    mut_escape_site_summary_df,
    replicate_data=False,
    addtl_tooltip_stats=None,
    epitope_colors,
    epitopes=None,
    sites=None,
    all_sites=True,
    share_ylims=True,
    height=100,
    width=900,
    init_metric="total positive",
    zoom_bar_width=500,
    avg_to_plot="escape_mean",
):
    r"""Line plots of mutation escape :math:`\beta_{m,e}` at each site.

    Parameters
    -----------
    mut_escape_site_summary_df : pandas.DataFrame
        Site-level escape in format of
        :attr:`polyclonal.polyclonal.Polyclonal.mut_escape_site_summary_df`.
    replicate_data : bool
        `mut_escape_site_summary_df` is from a collections of replicates as from
        :class:`polyclonal.polyclonal_collection.PolyclonalCollection`.
    addtl_tooltip_stats : list or None
        Additional mutation-level stats to show in tooltip.
    epitope_colors : dict
        Maps each epitope name to its color.
    epitopes : array-like or None
        Make plots for these epitopes. If `None`, use all epitopes.
    sites : None or array-like
        If `None`, the sites in `mut_escape_site_summary_df` represent sequential
        integers (and any missing ones are filled in). If you specify something else
        here, it should be array-like giving sites to plot in order to plot them.
    all_sites : bool
        Plot all sites in range from first to last site even if some
        have no data. Has no meanining if ``sites`` is set.
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
    avg_to_plot : {"escape_mean", "escape_median"}
        If using `replicate_data` plot the escape mean or median.

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
    if replicate_data:
        escape_metrics = df["metric"].unique().tolist()
    else:
        escape_metrics = [
            m
            for m in df.columns
            if m not in {"epitope", "site", "wildtype", "n mutations"}
        ]

    if replicate_data:
        df = df.rename(columns={avg_to_plot: "escape"})[
            ["epitope", "site", "metric", "escape"]
        ]
    else:
        df = df.melt(
            id_vars=["epitope", "site"],
            value_vars=escape_metrics,
            var_name="metric",
            value_name="escape",
        )

    # do we have pre-site sites to sort?
    if (df["site"].dtype != int) and (sites is None):
        raise ValueError("input data frame has non-integer sites so set `sites`")
    if sites:
        if not set(df["site"]).issubset(sites):
            raise ValueError("sites in input data frame not subset of `sites`")
        # Because of this bug, cannot sort by sites completely:
        # https://github.com/altair-viz/altair/issues/2663
        # However, can sort up to ~1000 elements, which is usually enough that
        # regular sorting will do the rest (assuming <10,000 sites. So make a
        # `sort_sites` list that sorts the first 1000 elements and then check
        # that is enough. If that bug is fixed, then `sort_sites` can be removed
        # and the sorting can just be on sites.
        n_sort_sites = 1002  # this many does not raise error
        sort_sites = sites[:n_sort_sites]
        if list(sites) != [*sort_sites, *sorted(sites[n_sort_sites:])]:
            raise ValueError(
                f"Cannot sort {len(sites)=} non-integer due to this bug in altair:\n"
                "https://github.com/altair-viz/altair/issues/2663"
            )
    else:
        sort_sites = "ascending"

    if not sites:
        sites = df["site"].unique().tolist()
        if all_sites:
            sites = list(range(min(sites), max(sites) + 1))

    # fill any missing sites
    fill_df = pd.DataFrame(
        itertools.product(sites, epitopes, escape_metrics),
        columns=["site", "epitope", "metric"],
    )
    df = df.merge(fill_df, how="right")

    df = df.pivot_table(
        index=["site", "metric"], values="escape", columns="epitope", dropna=False
    ).reset_index()

    if init_metric not in set(df["metric"]):
        raise ValueError(f"invalid {init_metric=}\noptions: {df['metric'].unique()=}")

    metric_selection = alt.selection_point(
        fields=["metric"],
        bind=alt.binding_select(options=escape_metrics),
        name="escape",
        value=[{"metric": init_metric}],
    )

    line_selection = alt.selection_point(
        fields=["show_line"],
        bind=alt.binding_select(options=[True, False], name="show_line"),
        value=[{"show_line": True}],
    )

    zoom_brush = alt.selection_interval(
        encodings=["x"],
        mark=alt.BrushConfig(stroke="black", strokeWidth=2),
    )
    zoom_bar = (
        alt.Chart(df[["site"]].drop_duplicates())
        .mark_rect(color="gray")
        .encode(x=alt.X("site:O", sort=sort_sites))
        .add_parameter(zoom_brush)
        .properties(
            width=zoom_bar_width,
            height=15,
            title="site zoom bar",
        )
    )

    site_selector = alt.selection_point(
        on="mouseover",
        fields=["site"],
        empty=False,
    )

    # add error ranges
    if replicate_data:
        pivoted_error = (
            mut_escape_site_summary_df.pivot_table(
                index=["site", "metric"],
                columns="epitope",
                values="escape_std",
            )
            .reset_index()
            .rename(columns={epitope: f"{epitope} error" for epitope in epitopes})
        )
        df = df.merge(
            pivoted_error,
            on=["site", "metric"],
            how="left",
            validate="many_to_one",
        )
        for epitope in epitopes:
            df[f"{epitope} min"] = df[epitope] - df[f"{epitope} error"]
            df[f"{epitope} max"] = df[epitope] + df[f"{epitope} error"]
        # selection to show error bars
        df["error_bars"] = True
        error_bar_selection = alt.selection_point(
            fields=["error_bars"],
            init=[{"error_bars": True}],
            bind=alt.binding_select(options=[True, False], name="show_error"),
        )

    # add wildtypes
    df = df.merge(
        mut_escape_site_summary_df[["site", "wildtype"]].drop_duplicates(),
        how="left",
        on="site",
        validate="many_to_one",
    )

    # add any additional tooltips
    addtl_tooltips = []
    if addtl_tooltip_stats is not None:
        for c in addtl_tooltip_stats:
            if c not in mut_escape_site_summary_df.columns:
                raise ValueError(
                    f"`addtl_tooltip_stat` {c} not in `mut_escape_site_summary_df`"
                )
            df = df.merge(
                mut_escape_site_summary_df[["site", c]].drop_duplicates(),
                how="left",
                on="site",
                validate="many_to_one",
            )
            addtl_tooltips.append(
                alt.Tooltip(c, format=".2f")
                if mut_escape_site_summary_df[c].dtype == float
                else c
            )

    # add selection for minimum **absolute value** of metrics
    # first add column that gives the percent of the max for each metric and statistic
    df = (
        df.merge(
            df.melt(
                id_vars="metric",
                value_vars=epitopes,
                var_name="epitope",
                value_name="escape",
            )
            .assign(escape=lambda x: x["escape"].abs())
            .groupby("metric", as_index=False)
            .aggregate(_max=pd.NamedAgg("escape", lambda s: s.abs().max()))
        )
        .assign(percent_max=lambda x: 100 * x[epitopes].abs().max(axis=1) / x["_max"])
        .drop(columns="_max")
    )
    assert numpy.allclose(df["percent_max"].max(), 100), df["percent_max"].max()
    cutoff = alt.selection_point(
        fields=["percent_max_cutoff"],
        value=[{"percent_max_cutoff": 0}],
        bind=alt.binding_range(min=0, max=100, name="percent_max_cutoff"),
    )

    charts = []
    base_all = (
        alt.Chart(df)
        .transform_calculate(show_line="true")
        .encode(
            tooltip=[
                alt.Tooltip("site:O"),
                alt.Tooltip("wildtype:N"),
                *[alt.Tooltip(f"{epitope}:Q", format=".2f") for epitope in epitopes],
                *addtl_tooltips,
            ],
        )
    )
    for epitope in epitopes:
        base = base_all.encode(
            x=alt.X(
                "site:O",
                sort=sort_sites,
                title=("site" if epitope == epitopes[-1] else None),
                axis=(alt.Axis() if epitope == epitopes[-1] else None),
            ),
            y=alt.Y(f"{epitope}:Q", title="escape"),
        )
        # in case some sites missing values, we need foreground and background
        background = (
            base.transform_filter(f"isValid(datum['{epitope}'])")
            .mark_line(size=1, color=epitope_colors[epitope])
            .encode(opacity=alt.condition(line_selection, alt.value(1), alt.value(0)))
        )
        foreground = base.mark_line(size=1, color=epitope_colors[epitope]).encode(
            opacity=alt.condition(line_selection, alt.value(1), alt.value(0)),
        )
        foreground_circles = (
            base.mark_circle(opacity=1, color=epitope_colors[epitope])
            .encode(
                size=alt.condition(site_selector, alt.value(75), alt.value(25)),
                stroke=alt.condition(
                    site_selector, alt.value("black"), alt.value(None)
                ),
            )
            .add_parameter(cutoff, site_selector, line_selection)
        )
        if replicate_data:
            error_bars = base.encode(
                y=alt.Y(f"{epitope} min", title="escape"),
                y2=f"{epitope} max",
                opacity=alt.condition(
                    error_bar_selection,
                    alt.value(0.75),
                    alt.value(0),
                ),
            ).mark_errorbar(color="gray", thickness=1.5)
            combined = background + foreground + foreground_circles + error_bars
        else:
            combined = background + foreground + foreground_circles
        charts.append(
            combined.add_parameter(metric_selection)
            .transform_filter(metric_selection)
            .transform_filter(zoom_brush)
            .transform_filter(alt.datum.percent_max >= cutoff.percent_max_cutoff)
            .properties(
                title=alt.TitleParams(
                    f"{epitope} epitope", color=epitope_colors[epitope]
                ),
                width=width,
                height=height,
            )
        )
        if replicate_data:
            charts[-1] = charts[-1].add_parameter(error_bar_selection)

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
        .configure(padding={"left": 15, "top": 5, "right": 5, "bottom": 12})
        .configure_axis(grid=False, labelOverlap="parity")
        .configure_title(anchor="start", fontSize=14)
    )


def mut_escape_heatmap(
    *,
    mut_escape_df,
    alphabet,
    epitope_colors,
    epitopes=None,
    sites=None,
    stat="escape",
    error_stat=None,
    addtl_tooltip_stats=None,
    all_alphabet=True,
    floor_at_zero=True,
    share_heatmap_lims=True,
    cell_size=12,
    zoom_bar_width=500,
    init_min_times_seen=1,
    epitope_label_suffix=" epitope",
    diverging_colors=False,
    max_min_times_seen=None,
    addtl_slider_stats=None,
):
    r"""Heatmaps of the mutation escape values, :math:`\beta_{m,e}`.

    Parameters
    ----------
    mut_escape_df : pandas.DataFrame
        Mutation-level escape in format of
        :attr:`polyclonal.polyclonal.Polyclonal.mut_escape_df`. Any
        wildtype entries should be set to 0 or are auto-filled to 0.
    alphabet : array-like
        Alphabet letters (e.g., amino acids) in order to plot them.
    epitope_colors : dict
        Maps each epitope name to its color.
    epitopes : array-like or None
        Make plots for these epitopes. If `None`, use all epitopes.
    stat : str
        Statistic in `mut_escape_df` to plot as escape.
    sites : None or array-like
        If `None`, the sites in `mut_escape_df` are assumed to represent sequential
        integers (and any missing ones are filled in). If you specify something else
        here, it should be array-like giving sites to plot in order to plot them.
    error_stat : str or None
        Measure of error to display in tooltip.
    addtl_tooltip_stats : list or None
        Additional mutation-level stats to show in tooltip.
    all_alphabet : bool
        Plot all letters in the alphabet (e.g., amino acids) even if some
        have no data.
    floor_at_zero : bool
        Set lower limit for color scale to zero, even if there are negative
        escape values or if minimum is >0. If True, then the *percent_max_cutoff*
        slider is also floored the same way.
    share_heatmap_lims : bool
        If `True`, let all epitopes share the same limits in color scale.
        If `False`, scale each epitopes colors to the min and max stat
        values for that epitope.
    cell_size : float
        Size of cells in heatmap.
    zoom_bar_width : float
        Width of zoom bar
    init_min_times_seen : int
        Initial cutoff for minimum times a mutation must be seen slider. Slider
        only shown if 'times_seen' in `addtl_tooltip_stats`. Also used for calculating
        the percent max cutoff values.
    epitope_label_suffix : str
        Suffix epitope labels with this.q
    diverging_colors : bool
        If `False`, colors in ``epitope_colors`` are assumed to be the upper color for
        white-to-<color> scale. If `True`, they are instead diverging color schemes with
        0 as white. Valid diverging schemes: https://vega.github.io/vega/docs/schemes/
    max_min_times_seen : int or None
        Maximum value for min times seen slider, or `None` for default.
    addtl_slider_stats : None or dict
        If you want additional sliders, key by other numeric properties in
        ``addtl_tooltip_stats`` and value is initial setting.

    Returns
    -------
    altair.Chart
        Interactive heat maps.

    """
    if epitopes is None:
        epitopes = mut_escape_df["epitope"].unique().tolist()
    elif not set(epitopes).issubset(mut_escape_df["epitope"]):
        raise ValueError("invalid entries in `epitopes`")

    if stat not in mut_escape_df:
        raise ValueError(f"{stat=} not in {mut_escape_df.columns=}")

    # do we have pre-site sites to sort?
    if (mut_escape_df["site"].dtype != int) and (sites is None):
        raise ValueError("`mut_escape_df` has non-integer sites so set `sites`")
    if sites:
        if not set(mut_escape_df["site"]).issubset(sites):
            raise ValueError("sites in `mut_escape_df` not subset of `sites`")
        # Because of this bug, cannot sort by sites completely:
        # https://github.com/altair-viz/altair/issues/2663
        # However, can sort up to ~1000 elements, which is usually enough that
        # regular sorting will do the rest (assuming <10,000 sites. So make a
        # `sort_sites` list that sorts the first 1000 elements and then check
        # that is enough. If that bug is fixed, then `sort_sites` can be removed
        # and the sorting can just be on sites.
        n_sort_sites = 1002  # this many does not raise error
        sort_sites = sites[:n_sort_sites]
        if list(sites) != [*sort_sites, *sorted(sites[n_sort_sites:])]:
            raise ValueError(
                f"Cannot sort {len(sites)=} non-integer due to this bug in altair:\n"
                "https://github.com/altair-viz/altair/issues/2663"
            )
    else:
        sort_sites = "ascending"

    df = mut_escape_df.query("epitope in @epitopes")

    # get alphabet
    extrachars = set(df["mutant"]).union(set(df["wildtype"])) - set(alphabet)
    if extrachars:
        raise ValueError(
            "`mut_escape_df` has letters not in `alphabet`:\n" + str(extrachars)
        )
    if not all_alphabet:
        alphabet = [c for c in alphabet if c in set(df["mutant"]) + set(df["wildtype"])]

    # get labels for escape to show on tooltips, potentially with error
    if error_stat is not None:
        if error_stat not in df.columns:
            raise ValueError(f"{error_stat=} not in {df.columns=}")
        label_df = (
            df.assign(
                label=lambda x: x.apply(
                    lambda r: f"{r[stat]:.2f} +/- {r[error_stat]:.2f}",
                    axis=1,
                )
            )
            .pivot_table(
                index=["site", "mutant"],
                values="label",
                columns="epitope",
                aggfunc=lambda x: " ".join(x),
            )
            .rename(columns={e: f"{e}{epitope_label_suffix}" for e in epitopes})
        )
        escape_tooltips = [f"{e}{epitope_label_suffix}" for e in epitopes]
    else:
        escape_tooltips = [
            alt.Tooltip(e, format=".2f", title=f"{e}{epitope_label_suffix}")
            for e in epitopes
        ]

    # add wildtypes not already in data frame, setting effects to 0
    existing_wts = set(df.query("mutant == wildtype")["site"])
    if any(df.query("mutant == wildtype")[stat] != 0):
        raise ValueError("some sites have wildtype != 0")
    all_wts = mut_escape_df.set_index("site")["wildtype"].to_dict()
    missing_wts = {r: wt for r, wt in all_wts.items() if r not in existing_wts}
    df = df.pivot_table(
        index=["site", "mutant"], values=stat, columns="epitope"
    ).reset_index()
    if missing_wts:
        missing_wt_df = (
            pd.Series(missing_wts)
            .rename_axis("site")
            .to_frame("mutant")
            .reset_index()
            .assign(**{e: 0 for e in epitopes})
        )
    if error_stat is not None:
        df = df.merge(
            label_df, how="left", on=["site", "mutant"], validate="one_to_one"
        )
        if missing_wts:
            missing_wt_df = missing_wt_df.assign(
                **{f"{e} epitope": "0" for e in epitopes}
            )
    if len(missing_wts):
        df = pd.concat([df, missing_wt_df], ignore_index=True)

    # add mutations and indicate which sites are wildtype
    df = df.assign(
        mutation=lambda x: (
            x["site"].map(all_wts) + x["site"].astype(str) + x["mutant"]
        ),
        is_wildtype=lambda x: x["site"].map(all_wts) == x["mutant"],
    )

    # zoom bar to put at top
    zoom_brush = alt.selection_interval(
        encodings=["x"], mark=alt.BrushConfig(stroke="black", strokeWidth=2)
    )
    zoom_bar = (
        alt.Chart(df[["site"]].drop_duplicates())
        .mark_rect(color="gray")
        .encode(x=alt.X("site:O", sort=sort_sites))
        .add_parameter(zoom_brush)
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
        for c in addtl_tooltip_stats:
            if c != "times_seen":
                add_tooltips.append(
                    alt.Tooltip(c, format=".2g")
                    if mut_escape_df[c].dtype == float
                    else c
                )
            else:
                # we must handle 'times_seen' differently to make it work with slider
                # for wildtype values, where it is assigned a dummy value below
                add_tooltips.append("times seen")

    if addtl_tooltip_stats is not None and "times_seen" in addtl_tooltip_stats:
        # set times seen for wildtype to max at site for slider to work
        site_max_times_seen = df.groupby("site")["times_seen"].max()
        df = df.assign(
            times_seen=lambda x: x.apply(
                lambda r: site_max_times_seen[r["site"]]
                if r["is_wildtype"]
                else r["times_seen"],
                axis=1,
            )
        )
        # note "times seen" is value in tooltip, "times_seen" for slider
        if (df["times_seen"] == df["times_seen"].astype(int)).all():
            df["times seen"] = df["times_seen"].astype(int).astype(str)
        else:
            df["times seen"] = df["times_seen"].map(lambda x: f"{x:.1f}")
        df["times seen"] = df["times seen"].where(~df["is_wildtype"], "na")
        # make time_seen selection slider, default
        # max is greater of median or mean across variants
        if max_min_times_seen is None:
            max_min_times_seen = math.ceil(
                max(
                    df["times_seen"].median(),
                    df["times_seen"].mean(),
                    init_min_times_seen,
                )
            )
        times_seen_cutoff = alt.selection_point(
            fields=["times_seen"],
            value=[{"times_seen": init_min_times_seen}],
            bind=alt.binding_range(
                min=math.floor(df["times_seen"].min()),
                max=max_min_times_seen,
                step=1,
                name="min_times_seen",
            ),
        )
    else:
        times_seen_cutoff = None

    addtl_sliders = {}
    if addtl_slider_stats is not None:
        for slider_stat, init_slider_stat in addtl_slider_stats.items():
            if not (addtl_tooltip_stats and slider_stat in addtl_tooltip_stats):
                raise ValueError(
                    f"addtl_slider_stat {slider_stat} not in addtl_tooltip_stats"
                )
            assert slider_stat in df.columns
            # set stat for wildtype to max at site for slider to retain wildtype
            stat_max = df[slider_stat].max()
            df[slider_stat] = df[slider_stat].where(~df["is_wildtype"], stat_max)
            # add slider
            addtl_sliders[slider_stat] = alt.selection_point(
                fields=[slider_stat],
                value=[{slider_stat: init_slider_stat}],
                name=slider_stat,
                bind=alt.binding_range(
                    min=df[slider_stat].min(),
                    max=df[slider_stat].max(),
                    step=1,
                    name=slider_stat,
                ),
            )

    # select cells
    cell_selector = alt.selection_point(on="mouseover", empty=False)

    # add selection for minimum  **absolute value** of escape maxed across site.
    # we do the calculation just for mutations at init_min_times_seen times seen
    if "times_seen" in df.columns:
        df_percent_max = df.query("times_seen >= @init_min_times_seen")
    else:
        df_percent_max = df
    if floor_at_zero:
        # the next line making a copy is needed to prevent pandas SettingWithCopyWarning
        df_percent_max = df_percent_max.copy()
        for e in epitopes:
            df_percent_max[e] = df_percent_max[e].clip(lower=0)
    _max = df_percent_max[epitopes].max().max()
    _min = df_percent_max[epitopes].min().min()
    df_percent_max = (
        df_percent_max.assign(_epitope_max=lambda x: x[epitopes].max(axis=1))
        .groupby("site", as_index=False)
        .aggregate(_site_max=pd.NamedAgg("_epitope_max", "max"))
        .assign(
            percent_max=lambda x: (100 * (x["_site_max"] - _min) / (_max - _min)).clip(
                lower=0
            )
        )
        .drop(columns="_site_max")
    )
    df = df.merge(df_percent_max, on="site", how="left", validate="many_to_one").assign(
        percent_max=lambda x: x["percent_max"].fillna(0)
    )
    assert numpy.allclose(df["percent_max"].max(), 100), df["percent_max"].max()
    percent_max_cutoff = alt.selection_point(
        fields=["percent_max_cutoff"],
        value=[{"percent_max_cutoff": 0}],
        bind=alt.binding_range(min=0, max=100, name="percent_max_cutoff"),
    )

    # make list of heatmaps for each epitope
    charts = [zoom_bar]
    # base chart
    base = alt.Chart(df).encode(
        x=alt.X("site:O", sort=sort_sites),
        y=alt.Y("mutant:O", sort=alphabet, scale=alt.Scale(domain=alphabet)),
    )
    # for inexplicable reason, this dummy chart is needed below for coloring to work
    dummy = base.mark_rect(opacity=0).encode(color=alt.Color("dummy:N", legend=None))
    # wildtype marks
    wildtype = base.transform_filter(alt.datum.is_wildtype).mark_text(
        color="black",
        text="x",
    )
    # now make heatmaps
    for epitope in epitopes:
        if share_heatmap_lims:
            vals = df[list(epitopes)].values
        else:
            vals = df[epitope].values
        escape_max = numpy.nanmax(vals)
        if floor_at_zero:
            escape_min = 0
        else:
            escape_min = numpy.nanmin(vals)
        if not (escape_min < escape_max):
            raise ValueError("escape min / max do not span a valid range")
        heatmap = base.mark_rect().encode(
            color=alt.Color(
                epitope,
                type="quantitative",
                # diverging color scales: https://stackoverflow.com/a/70296527
                scale=(
                    alt.Scale(domainMid=0, scheme=epitope_colors[epitope])
                    if diverging_colors
                    else alt.Scale(
                        range=color_gradient_hex("white", epitope_colors[epitope], 20),
                        type="linear",
                        domain=(escape_min, escape_max),
                        clamp=True,
                    )
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
            tooltip=["mutation"] + escape_tooltips + add_tooltips,
        )
        # combine the elements
        charts.append(
            (heatmap + dummy + wildtype)
            .add_parameter(cell_selector, percent_max_cutoff)
            .transform_filter(zoom_brush)
            .transform_filter(
                alt.datum.percent_max >= percent_max_cutoff.percent_max_cutoff
            )
            .properties(
                title=alt.TitleParams(
                    f"{epitope}{epitope_label_suffix}",
                    color="black" if diverging_colors else epitope_colors[epitope],
                ),
                width={"step": cell_size},
                height={"step": cell_size},
            )
        )
        if times_seen_cutoff:
            charts[-1] = (
                charts[-1]
                .add_parameter(times_seen_cutoff)
                .transform_filter(alt.datum.times_seen >= times_seen_cutoff.times_seen)
            )
        for slider_stat, slider in addtl_sliders.items():
            charts[-1] = (
                charts[-1]
                .add_parameter(slider)
                .transform_filter(f"datum.{slider_stat} >= {slider_stat}.{slider_stat}")
            )

    chart = (
        alt.vconcat(
            *charts,
            spacing=0,
        )
        .configure_axis(labelOverlap="parity")
        .configure_title(anchor="start", fontSize=14)
        .configure_view(fill="gray", fillOpacity=0.25)
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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
