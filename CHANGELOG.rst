=========
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com>`_.

2.0
---
Many of these changes are **backward incompatible** with respect to plotting.

- Changed plotting of escape. Replaced ``mut_escape_heatmap`` and ``mut_escape_lineplot`` with ``lineplot_and_heatmap``
- Changed default epitope colors for ``Polyclonal`` objects.
- Replaced ``Polyclonal.mut_escape_heatmap`` and ``Polyclonal.mut_escape_lineplot`` with ``Polyclonal.mut_escape_plot``.
- Replaced ``PolyclonalCollection.mut_escape_heatmap`` and ``PolyclonalCollection.mut_escape_lineplot`` with ``PolyclonalCollection.mut_escape_plot``.

1.2
----
- ``PolyclonalCollection`` plotting (specifically lineplot) still works even if there is just one model in collection. Before this edge case caused an error.
- Allow negative site numbers.

1.1
----
- Sort sites in mutation and site-escape data frames output by ``PolyclonalCollection``.

1.0
---------------------------
- Renamed the bootstrapping models from ``PolyclonalCollection`` to ``PolyclonalBootstrap`` and made ``PolyclonalCollection`` a general-purpose class for collection of ``Polyclonal`` objects. This makes the idea of model collections more general, and better aligns the class names with what they actually do. This is a **backward-incompatible change**.
- Added methods for correlating models to ``PolyclonalCollection``: methods ``mut_escape_corr``, ``mut_escape_corr_heatmap``.
- Added ``sites`` parameter to ``Polyclonal`` to enable non-sequential-integer (eg, reference) based site numbering, and propagated this change to plotting and ``PolyclonalCollection``.
- Added the ``PolyclonalAverage`` class for computing the average of models.
- ``PolyclonalCollection`` and subclasses return both mean and median and provide option to plot either, and ``PolyclonalCollection.default_avg_to_plot`` attribute added.
- Remove the old ``Polyclonal.harmonize_epitopes_with`` method that modified ``Polyclonal`` models in place, and replaced with the new ``Polyclonal.epitope_harmonized_model`` that returns a copy of the original model with the epitopes harmonized and also provides guarantees about consistent epitope order, etc. This is a **backward-incompatible change**.
- Added ``alphabets.biochem_order_aas`` and by default plot heatmaps with biochemically ordered amino acids.
- Added `n_replicates` slider to ``PolyclonalCollection.mut_escape_heatmap``
- Added `min_replicates` to ``PolyclonalCollection.mut_escape_lineplot``
- Added ``plot.corr_heatmap`` function.
- Added ``utils.tidy_to_corr`` function.
- Added ``polyclonal_collection.fit_models`` to fit multiple models using multiprocessing.
- Added ``RBD_average.ipynb`` notebook.
- Fix bug in setting ``epitope_colors`` as dict in ``Polyclonal``.
- Fix ``SettingWithCopyWarning`` in heatmap plotting.
- Added ``letter_suffixed_sites`` parameter to ``MutationParser`` and ``site_level_variants``.
- Added ``sites`` to ``plot.mut_escape_heatmap`` and ``plot.mut_escape_lineplot`` to enable ordering of str site numbers and utilized within ``Polyclonal`` objects.
- Increment ``binarymap`` version requirement to >= 0.5.
- Changed real antibody example from REGN10933 to LY-CoV1404

0.3
---------------------------
- Add some options that generalize ``mut_escape_heatmap``, specifically:
  * allow wildtype entries to be specified (with effects of 0) in ``mut_escape_df``. This helps allow additional tooltips.
  * added ``max_min_times_seen`` and some tweaks to ``times seen`` tooltip
  * added ``addtl_sliders_stats``

0.2
---------------------------
- Add some options that generalize ``mut_escape_heatmap``, specifically:
  * added ``epitope_label_suffix`` parameter
  * added ``diverging_colors`` parameter
  * changed ``percent_max_cutoff`` slider to work on real rather than absolute values and with non-zero minimum values.

0.1
---------------------------
Initial release

