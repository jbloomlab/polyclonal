=========
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com>`_.

3.4
---
- Plotting improvements:
 - Add ``heatmap_max_fixed`` and ``heatmap_min_fixed`` to ``plot.lineplot_and_heatmap``
- Make it easier to access per-model measurements for averages of models:
 - added ``PolyclonalCollection.unique_descriptor_names`` attribute.
 - added ``PolyclonalCollection.mut_escape_df_model_values`` property
 - added ``per_model_tooltip`` option to ``PolyclonalCollection.mut_escape_plot``, and make this the default when <=5 models.

3.3
---
- Add options for lineplot only or heatmap only in ``plot.lineplot_and_heatmap`` (these are ``show_zoombar``, ``show_lineplot``, and ``show_heatmap``).
- Add ``scale_stat_col`` option to ``plot.lineplot_and_heatmap``.
- Add ``rename_stat_col`` option to ``plot.lineplot_and_heatmap``.

3.2
---
- Allow non-integer PDB numbers in B-factor re-assignment in ``reassign_b_factor``.

3.1
----
- Change default positive colors.

3.0
----
- Make ``Polyclonal.spatial_distances`` a public attribute.
- ``Polyclonal.fit`` allows epitopes with identical activities if they have different escape.
- Activity regularization penalizes both positive and negative values.
- Checks and int versus str dtype adjustment for ``spatial_distances`` in ``Polyclonal``.
- Adjust activity regularization based on concentration geometric mean so it is not sensitive to units of concentration.
- Renamed what was previously called the epitope similarity regularization to epitope uniqueness-squared, as it's a second uniqueness regularization but operates on square of escape values.
- Change default regularization strengths in ``Polyclonal.fit``.
- Rename ``reg_spatial_weight2`` to ``reg_spatial2_weight`` for ``Polyclonal.fit``.
- Update default values of regularization weights.
- Added antibody cocktail and HIV serum examples.
- Updated examples, for instance by adding spatial regularization to RBD example and slightly changing simulated data.

2.6
------
- Flatten mutation-escape values for RBD simulation so most values are roughly zero. Previously many values were ~0.5 rather than 0. Also adjust activities for this simulation.
- Improve column spacing when ``Polyclonal.fit`` prints log.
- Added 'atom' as output column in ``pdb_utils.extract_atom_locations``
- Added ``pdb_utils.inter_residue_distances``
- Added ``Polyclonal.distance_matrix`` attribute, set via ``spatial_distances`` parameter.
- Added spatial regularization to ``Polyclonal.fit``
- Added uniqueness regularization to ``Polyclonal.fit`` as an alternative to similarity regularization that does not go with square of site-level values.

2.5
---
- Add epitope similarity regularization that can be tuned by the parameter ``reg_similarity_weight``.
- Add ``real_mAb_cocktail.ipynb`` notebook that fits model to a real mAb cocktail dataset. 

2.4
---
- Fix bug introduced in version 2.3 that dropped wildtype sites if there were minimums set in ``slider_binding_range_kwargs`` to ``lineplot_and_heatmap``.

2.3
---
- ``lineplot_and_heatmap`` computes the limit for the heatmap range **after** applying the minimum filters specified in the filters. This avoids having the range determined by mutations that are never plotted, and so is sort of a bug fix (prior behavior wasn't strictly a bug, but did not give sensible behavior).

2.2
---
- Require at least ``pandas`` 1.5.
- Some minor changes to avoid ``pandas`` warnings about future deprecations.
- For ``Polyclonal`` initialization, allow `data_mut_escape_overlap` != "exact" even with `sites` set

2.1
---
- ``lineplot_and_heatmap`` filters site max value after other slider filters, this gives correct behavior and is sort of a bug fix for the plots.
- Add ``slider_binding_range_kwargs`` to ``lineplot_and_heatmap``
- Allow ``df_to_merge`` to be list for ``mut_escape_plot`` methods.

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

