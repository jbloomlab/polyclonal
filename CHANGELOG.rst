=========
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com>`_.


1.0
---------------------------
- Renamed the bootstrapping models from ``PolyclonalCollection`` to ``PolyclonalBootstrap`` and made ``PolyclonalCollection`` a general-purpose class for collection of ``Polyclonal`` objects. This makes the idea of model collections more general, and better aligns the class names with what they actually do. This is a **backward-incompatible change**.
- Added methods for correlating models to ``PolyclonalCollection``: methods ``mut_escape_corr``
- Added the ``PolyclonalAverage`` class for computing the average of models.
- Remove the old ``Polyclonal.harmonize_epitopes_with`` method that modified ``Polyclonal`` models in place, and replaced with the new ``Polyclonal.epitope_harmonized_model`` that returns a copy of the original model with the epitopes harmonized and also provides guarantees about consistent epitope order, etc. This is a **backward-incompatible change**.
- Added ``plot.corr_heatmap`` function.
- Added ``utils.tidy_to_corr`` function.
- Added ``polyclonal_collection.fit_models`` to fit multiple models using multiprocessing.
- Added ``RBD_average.ipynb`` notebook.
- Fix bug in setting ``epitope_colors`` as dict in ``Polyclonal``.

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

