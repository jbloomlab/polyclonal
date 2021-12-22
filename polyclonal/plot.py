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


TAB10_COLORS_NOGRAY = tuple(c for c in
                            matplotlib.colors.TABLEAU_COLORS.values()
                            if c != '#7f7f7f')
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
                    name='_',
                    colors=[start, end],
                    N=n)
    return [matplotlib.colors.rgb2hex(tup) for tup in cmap(list(range(0, n)))]


def activity_wt_barplot(*,
                        activity_wt_df,
                        epitope_colors,
                        epitopes=None,
                        width=110,
                        height_per_bar=25,
                        ):
    r"""Bar plot of activity against each epitope, :math:`a_{\rm{wt},e}`.

    Parameters
    ----------
    activity_wt_df : pandas.DataFrame
        Epitope activities in format of
        :attr:`polyclonal.Polyclonal.activity_wt_df`.
    epitope_colors : dict
        Maps each epitope name to its color.
    epitopes : array-like or None
        Include these epitopes in this order. If `None`, use all epitopes
        in order found in ``activity_wt_df``.
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
        epitopes = activity_wt_df['epitope'].tolist()
    elif not set(epitopes).issubset(activity_wt_df['epitope']):
        raise ValueError('invalid entries in `epitopes`')
    df = (activity_wt_df
          .query('epitope in @epitopes')
          .assign(epitope=lambda x: pd.Categorical(x['epitope'],
                                                   epitopes,
                                                   ordered=True)
                  )
          .sort_values('epitope')
          )

    barplot = (
        alt.Chart(df)
        .encode(x='activity:Q',
                y='epitope:N',
                color=alt.Color(
                   'epitope:N',
                   scale=alt.Scale(domain=epitopes,
                                   range=[epitope_colors[e]
                                          for e in epitopes]),
                   legend=None,
                   ),
                tooltip=[alt.Tooltip('epitope:N'),
                         alt.Tooltip('activity:Q', format='.3g')],
                )
        .mark_bar(size=0.75 * height_per_bar)
        .properties(width=width,
                    height={'step': height_per_bar})
        .configure_axis(grid=False)
        )

    return barplot


def mut_escape_lineplot(*,
                        mut_escape_site_summary_df,
                        epitope_colors,
                        epitopes=None,
                        all_sites=True,
                        share_ylims=True,
                        height=100,
                        width=900,
                        init_metric='mean',
                        zoom_bar_width=500,
                        ):
    r"""Line plots of mutation escape :math:`\beta_{m,e}` at each site.

    Parameters
    -----------
    mut_escape_site_summary_df : pandas.DataFrame
        Site-level escape in format of
        :attr:`polyclonal.Polyclonal.mut_escape_site_summary_df`.
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
        Metric to show initially (others can be selected by dropdown).
        One of metrics in :attr:`Polyclonal.site_summary_df`.
    zoom_bar_width : float
        Width of zoom bar

    Returns
    -------
    altair.Chart
        Interactive plot.

    """
    if epitopes is None:
        epitopes = mut_escape_site_summary_df['epitope'].unique().tolist()
    elif not set(epitopes).issubset(mut_escape_site_summary_df['epitope']):
        raise ValueError('invalid entries in `epitopes`')

    df = mut_escape_site_summary_df.query('epitope in @epitopes')
    escape_metrics = [m for m in df.columns
                      if m not in {'epitope', 'site', 'wildtype'}]

    sites = df['site'].unique().tolist()
    if all_sites:
        sites = list(range(min(sites), max(sites) + 1))

    df = (df
          .merge(pd.DataFrame(itertools.product(sites, epitopes),
                              columns=['site', 'epitope']),
                 on=['site', 'epitope'], how='right')
          .sort_values('site')
          .melt(id_vars=['epitope', 'site', 'wildtype'],
                var_name='metric',
                value_name='escape'
                )
          .pivot_table(index=['site', 'wildtype', 'metric'],
                       values='escape',
                       columns='epitope',
                       dropna=False)
          .reset_index()
          )

    y_axis_dropdown = alt.binding_select(options=escape_metrics)
    y_axis_selection = alt.selection_single(fields=['metric'],
                                            bind=y_axis_dropdown,
                                            name='escape',
                                            init={'metric': init_metric})

    zoom_brush = alt.selection_interval(encodings=['x'],
                                        mark=alt.BrushConfig(
                                            stroke='black',
                                            strokeWidth=2),
                                        )
    zoom_bar = (alt.Chart(df)
                .mark_rect(color='gray')
                .encode(x='site:O')
                .add_selection(zoom_brush)
                .properties(width=zoom_bar_width,
                            height=15,
                            title='site zoom bar',
                            )
                )

    site_selector = alt.selection(type='single',
                                  on='mouseover',
                                  fields=['site'],
                                  empty='none')

    charts = []
    for epitope in epitopes:
        base = (
            alt.Chart(df)
            .encode(x=alt.X('site:O',
                            title=('site' if epitope == epitopes[-1]
                                   else None),
                            axis=(alt.Axis() if epitope == epitopes[-1]
                                  else None),
                            ),
                    y=alt.Y(epitope,
                            type='quantitative',
                            title='escape',
                            scale=alt.Scale(),
                            ),
                    tooltip=[alt.Tooltip('site:O'),
                             alt.Tooltip('wildtype:N'),
                             *[alt.Tooltip(f"{epitope}:Q", format='.3g')
                               for epitope in epitopes]
                             ]
                    )
            )
        # in case some sites missing values, background thin transparent
        # over which we put darker foreground for measured points
        background = (
            base
            .transform_filter(f"isValid(datum['{epitope}'])")
            .mark_line(opacity=0.5, size=1,
                       color=epitope_colors[epitope])
            )
        foreground = (
            base
            .mark_line(opacity=1, size=1.5,
                       color=epitope_colors[epitope])
            )
        foreground_circles = (
            base
            .mark_circle(opacity=1,
                         color=epitope_colors[epitope])
            .encode(size=alt.condition(site_selector,
                                       alt.value(75),
                                       alt.value(25)),
                    stroke=alt.condition(site_selector,
                                         alt.value('black'),
                                         alt.value(None)),
                    )
            .add_selection(site_selector)
            )
        charts.append((background + foreground + foreground_circles)
                      .add_selection(y_axis_selection)
                      .transform_filter(y_axis_selection)
                      .transform_filter(zoom_brush)
                      .properties(
                            title=alt.TitleParams(
                                      f"{epitope} epitope",
                                      color=epitope_colors[epitope]),
                            width=width,
                            height=height)
                      )

    return (alt.vconcat(zoom_bar,
                        (alt.vconcat(*charts, spacing=10)
                         .resolve_scale(y='shared' if share_ylims
                                        else 'independent')
                         ),
                        spacing=10)
            .configure_axis(grid=False, labelOverlap='parity')
            .configure_title(anchor='start', fontSize=14)
            )


def mut_escape_heatmap(*,
                       mut_escape_df,
                       alphabet,
                       epitope_colors,
                       epitopes=None,
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
        :attr:`polyclonal.Polyclonal.mut_escape_df`.
    alphabet : array-like or None
        Alphabet letters (e.g., amino acids) in order to plot them.
    epitope_colors : dict
        Maps each epitope name to its color.
    epitopes : array-like or None
        Make plots for these epitopes. If `None`, use all epitopes.
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
        If `False`, scale each epitopes colors to the min and max escape
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
        epitopes = mut_escape_df['epitope'].unique().tolist()
    elif not set(epitopes).issubset(mut_escape_df['epitope']):
        raise ValueError('invalid entries in `epitopes`')

    df = mut_escape_df.query('epitope in @epitopes')

    # get alphabet and sites, expanding to all if needed
    extrachars = set(df['mutant']).union(set(df['wildtype'])) - set(alphabet)
    if extrachars:
        raise ValueError('`mut_escape_df` has letters not in `alphabet`:\n' +
                         str(extrachars))
    if not all_alphabet:
        alphabet = [c for c in alphabet if c in set(df['mutant']) +
                    set(df['wildtype'])]
    sites = df['site'].unique().tolist()
    if all_sites:
        sites = list(range(min(sites), max(sites) + 1))

    wts = mut_escape_df.set_index('site')['wildtype'].to_dict()

    df = (df
          [['epitope', 'site', 'mutant', 'escape']]
          .pivot_table(index=['site', 'mutant'],
                       values='escape',
                       columns='epitope')
          .reset_index()
          .merge(pd.DataFrame(itertools.product(sites, alphabet),
                              columns=['site', 'mutant']),
                 how='right')
          .assign(wildtype=lambda x: x['site'].map(wts),
                  mutation=lambda x: (x['wildtype'].fillna('') +
                                      x['site'].astype(str) + x['mutant']),
                  mutant=lambda x: pd.Categorical(x['mutant'], alphabet,
                                                  ordered=True),
                  # mark wildtype cells with a `x`
                  wildtype_char=lambda x: (x['mutant'] == x['wildtype']
                                           ).map({True: 'x', False: ''}),
                  )
          .sort_values(['site', 'mutant'])
          )
    # wildtype has escape of 0 by definition
    for epitope in epitopes:
        df[epitope] = df[epitope].where(df['mutant'] != df['wildtype'], 0)

    # zoom bar to put at top
    zoom_brush = alt.selection_interval(encodings=['x'],
                                        mark=alt.BrushConfig(
                                                    stroke='black',
                                                    strokeWidth=2)
                                        )
    zoom_bar = (alt.Chart(df)
                .mark_rect(color='gray')
                .encode(x='site:O')
                .add_selection(zoom_brush)
                .properties(width=zoom_bar_width,
                            height=15,
                            title='site zoom bar',
                            )
                )

    # select cells
    cell_selector = alt.selection_single(on='mouseover',
                                         empty='none')

    # make list of heatmaps for each epitope
    charts = [zoom_bar]
    for epitope in epitopes:
        # base chart
        base = (alt.Chart(df)
                .encode(x=alt.X('site:O'),
                        y=alt.Y('mutant:O',
                                sort=alt.EncodingSortField(
                                            'y',
                                            order='ascending')
                                ),
                        )
                )
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
            raise ValueError('escape min / max do not span a valid range')
        heatmap = (base
                   .mark_rect()
                   .encode(
                       color=alt.Color(
                            epitope,
                            type='quantitative',
                            scale=alt.Scale(
                               range=color_gradient_hex(
                                   'white', epitope_colors[epitope], 10),
                               type='linear',
                               domain=(escape_min, escape_max),
                               clamp=True,
                               ),
                            legend=alt.Legend(orient='left',
                                              title='gray is n.d.',
                                              titleFontWeight='normal',
                                              gradientLength=100,
                                              gradientStrokeColor='black',
                                              gradientStrokeWidth=0.5)
                            ),
                       stroke=alt.value('black'),
                       strokeWidth=alt.condition(cell_selector,
                                                 alt.value(2.5),
                                                 alt.value(0.2)),
                       tooltip=[alt.Tooltip('mutation:N')] +
                               [alt.Tooltip(f"{epitope}:Q", format='.3g')
                                for epitope in epitopes],
                       )
                   )
        # nulls for cells with missing data
        nulls = (base
                 .mark_rect()
                 .transform_filter(f"!isValid(datum['{epitope}'])")
                 .mark_rect(opacity=0.25)
                 .encode(alt.Color('escape:N',
                                   scale=alt.Scale(scheme='greys'),
                                   legend=None),
                         )
                 )
        # mark wildtype cells
        wildtype = (base
                    .mark_text(color='black')
                    .encode(text=alt.Text('wildtype_char:N'))
                    )
        # combine the elements
        charts.append((heatmap + nulls + wildtype)
                      .interactive()
                      .add_selection(cell_selector)
                      .transform_filter(zoom_brush)
                      .properties(
                            title=alt.TitleParams(
                                    f"{epitope} epitope",
                                    color=epitope_colors[epitope]),
                            width={'step': cell_size},
                            height={'step': cell_size})
                      )

    return (alt.vconcat(*charts,
                        spacing=0,
                        )
            .configure_axis(labelOverlap='parity')
            .configure_title(anchor='start', fontSize=14)
            )


if __name__ == '__main__':
    import doctest
    doctest.testmod()
