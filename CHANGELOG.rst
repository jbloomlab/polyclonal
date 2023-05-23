=========
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com>`_.

6.0
---
- Upgrade minimum ``altair`` to 5.0.0
- Use ``ruff`` as linter rather than ``flake8``
- Add ``Polyclonal.mut_icXX_df`` method
- Add ``Polyclonal.mut_icXX_plot`` method
- Add ``PolyclonalCollection.mut_icXX_df_replicates``, ``PolyclonalCollection.mut_icXX_df``, and ``PolyclonalCollection.mut_icXX_df_w_model_values`` methods.
- Add ``PolyclonalCollection.mut_icXX_plot`` method

5.3
---
- Add *min_escape_magnitude* as an option for ``PolyclonalAverage`` escape averages and plots. This gives the value across models with the lowest magnitude (smallest absolute value).

5.2
---
- In ``polyclonal.plot.lineplot_and_heatmap``, apply any hiding filter sliders before doing the max of the stat at the site slider. This avoids showing sites with all hidden values when trying to look at max.
- Pass ``nbval`` tests with ``pandas`` 2.0.

5.1
---
- Update to using ``altair`` version 5.0.0rc1, which can be installed by `pip`. This also means for the first time ``polyclonal`` itself can be on PyPI, which changes installation instructions.

5.0
---
- Increase default ``reg_activity_weight`` from 1.0 to 2.0, note that this will change results relative to models fit with earlier versions with the old default weight.
- Improvements to fitting of models:
  - The optimization bounds to ``Polyclonal.fit`` can now be set as parameters (``activity_bounds``, etc) rather than hard-coded.
  - Change regularization on Hill coefficient to a more quadratic form.
  - Adjust regularization weight for Hill coefficient (decreasing to 25)
  - Put upper bound on non-neutralizable fraction of 0.5.
  - Add (and activate by default) ``fit_fixed_first`` to fit a model with a fixed Hill coefficient and non-neutralized fraction first, and the ``fit_fixed_first_reg_activity_weight`` option to fit it with a higher weight.
- Add ``check_concentration_scale`` to ``Polyclonal`` to keep concentrations in data to fit in reasonable range.

4.1
----
- Added ``sites_to_show`` option to ``polyclonal.plot.lineplot_and_heatmap``.
- Remove `is_weighted_replicate` from ``curves_plot`` tooltip (it was an error this was ever shown).

4.0
---
- Enable (and activate by default) fitting of a Hill coefficient and non-neutralized fraction in the curves. Previously, the Hill coefficient had been constrained to one and the non-neutralized fraction to zero. This is a **major change** that will alter the results of fitting models. To get the old behavior, call ``Polyclonal.fit`` with ``fix_hill_coefficient=True`` and ``fix_non_neutralized_frac=True``. Specific changes associated with this update include:
  - Add ``hill_coefficient_df`` and ``non_neutralized_frac_df`` as parameters and properties of ``Polyclonal`` and ``PolyclonalCollection``.
  - Restructure internal organization of ``Polyclonal._params`` (this was private, so external code should not be using anyway).
  - Add regularization to Hill coefficient and non-neutralized fraction to ``Polyclonal.fit``.
  - Add the "two-epitope" example to illustrate the non-one Hill coefficient and non-neutralized fractions not equal to zero.
- Update ``reg_escape_weight`` to 0.05 in ``Polyclonal.fit`` because it seems like many people in lab were using larger values. This will **change results** of fitting models because old default was 0.02.
- Add the ``curves_plot`` and ``curve_specs_df`` to ``Polyclonal`` / ``PolyclonalCollection``,  ``curves_plot`` to the ``plots`` module, and prefer use of this over the activity barplots in the docs. The reason is that this shows the activity, Hill coefficient, and non-neutralized frac.
- Remove bootstrapping from docs as this isn't really a recommended procedure.
- Updated the default colors for epitopes (changing fourth to light blue and adding olive as fifth)

3.4
---
- Plotting improvements:
 - Add ``heatmap_max_fixed`` and ``heatmap_min_fixed`` to ``plot.lineplot_and_heatmap``
- Make it easier to access per-model measurements for averages of models:
 - added ``PolyclonalCollection.unique_descriptor_names`` attribute.
 - added ``PolyclonalCollection.mut_escape_df_model_values`` property
 - added ``per_model_tooltip`` option to ``PolyclonalCollection.mut_escape_plot``, and make this the default when <=5 models.
- Heatmap tooltips now show missing (null) numerical values as *NaN* rather than as 0.
- Update ``altair`` version to commit to *f8912bad75d4247ab726b639968b13315161660a* (note that the latest version still not merged on ``altair`` main branch, so still having to install from GitHub). In this new version, ``alt.add_parameter`` becomes ``alt.add_params`` and ``alt.parameter`` becomes ``alt.param``.
- Implemented slider that hides rather than filters mutations on the heatmaps. This is designed for mutation effect filtering where we'd like to be able to see which mutations have poor effects. They are now filtered from lineplot calculation and shown as gray. Adds the following parameters to ``plot.lineplot_and_heatmap``:
 - ``addtl_slider_stats_hide_not_filter``
 - ``hide_color``
 - also very slight changes to colors in background of heatmap.
- Re-order default positive color scheme to put green before dark byzantium to give better clarity relative to hidden / filtered values in heatmaps.
- Updated notebooks to use new plotting.

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

