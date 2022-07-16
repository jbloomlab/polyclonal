=========
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com>`_.


1.0
---------------------------
- **Backward incompatible change:** renamed the bootstrapping models from ``PolyclonalCollection`` to ``PolyclonalBootstrap`` and made ``PolyclonalCollection`` a general-purpose class for collection of ``Polyclonal`` objects. This makes the idea of model collections more general, and better aligns the class names with what they actually do.
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

