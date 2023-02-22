Regularization
--------------

The prior examples used `regularization <https://en.wikipedia.org/wiki/Regularization_(mathematics)>`_ of the model parameters.
This is an absolutely crucial aspect of the fitting, as it helps ensure the model gives "sensibe" results that are not overfit.

The default values of the regularization weights are hopefully reasonable, but in many cases you may need to tune the regularization weights to give more sensible results.

Here we briefly summarize the regularization parameters and what they do (for more technical details, see `here <https://jbloomlab.github.io/polyclonal/optimization.html>`_.

All the regularization weights are set as parameters to `Polyclonal.fit <https://jbloomlab.github.io/polyclonal/polyclonal.polyclonal.html#polyclonal.polyclonal.Polyclonal.fit>`_.
Specifically:

 - **Mutation escape-value regularization** (``reg_escape_weight``): this weight determines how strongly we bias all the mutation-escape values to be zero. The idea behind this regularization is that most mutations do not mediate escape, and so if many many sites show escape that probably indicates noise or over-fitting that we want to regularizat away.

 - **Spread of escape a site regularization** (``reg_spread_weight``): this weight determines how strongly we bias all of the different mutations at a site to have the same escape. The idea behind this regularization is that *usually* (but not always) different mutations at a site will have similar effects on escape.

 - **Spatial regularization** (``reg_spatial_weight`` and ``reg_spatial_weight2``): these weights determine how strongly we bias each epitope to have sites that are proximal in three dimensional structure. Using this regularization requires that you provide the spatial distances between sites when initializing the ``Polyclonal`` object (via the ``spatial_distances`` parameter), otherwise these weights have no effects. The first weight penalizes the distance between sites in the same epitope with lots of escape, and the second weight penalizes the **square** of the distance between sites in the same epitope with lots of escape.

 - **Epitope uniqueness regularization** (``reg_uniqueness_weight`` and ``reg_uniqueness2_weight``): these weights penalize two different epitopes having strong escape at the same site, based on the notion epitopes should be largely unique. The first weight penalizes the product of average escape at a site, the second weight can be stronger as it penalizes the product of the **square** of escape at a site.

 - **Epitope activity regularization** (``reg_activity_weight``): this weight penalizes epitopes having very high or very low activity. To be precise, it penalizes how much the activities differ from the log of the geometric mean concentrations used for the input data (this will be zero if you have put your concentrations on a scale where the values are ~1). This can be quite useful to avoid both fitting too many and too few epitopes.
 
 - **Hill coefficient regularization** (``reg_Hill_coefficient``): this weight penalizes Hill coefficients :math:`n_e` that differ from one. To be precise, it symmetrically penalizes :math:`1 / n_e` for :math:`n_e < 1` and :math:`n_e` for :math:`n_e > 1`. This regularization should generally be large to ensure the Hill coefficient does not get wildly different from one.
 
 - **non-neutralized fraction regularization** (``reg_non_neutralized_frac``): this weight penalizes non-neutralizable fractions :math:`t_e` that differ from zero. Note that the optimization also uses bounds that constrain :math:`t_e \ge 0`. This regularization should be large to ensure the non-neutralizable fraction does not get dramatically larger than zero.

In addition to regularization, the ``fix_hill_coefficient`` and ``fix_non_neutralized_frac`` options to `Polyclonal.fit <https://jbloomlab.github.io/polyclonal/polyclonal.polyclonal.html#polyclonal.polyclonal.Polyclonal.fit>`_ allow you to fix the Hill coefficient and non-neutralized fractions to their initial values, which by default are one and zero.
Normally fitting rather than fixing these parameters should give a better fit to the data, but sometimes the fitting causes problems in optimization or yields unrealistic values, in which case fixing one or both of these parameters could be helpful.
