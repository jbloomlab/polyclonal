r"""Generate :math:`\beta_{m,e}` from deep mutational scanning data.

The deep mutational scanning measurements are in
``RBD_mutation_escape_fractions.csv``, and consist of estimates of the
“escape fraction” :math:`x_{m,e}` for each mutation :math:`m` against the
antibody targeting epitope :math:`e`. These escape fractions are the
probability that a RBD carrying only that amino-acid mutation is unbound by
the antibody at a concentration where only ~0.1% of the unmutated RBD is
unbound. We perform the calculations noting that
:math:`x_{m,e}` represents the :math:`U_e` values.

We also adjust values less than 1.2 be zero.

"""

import numpy

import pandas as pd

escape_frac_floor = 0.0001
escape_frac_ceil = 0.9999

antibody_to_epitope = {
    "LY-CoV016": "class 1",
    "LY-CoV555": "class 2",
    "REGN10987": "class 3",
}

mut_escape_df = pd.read_csv("RBD_mutation_escape_fractions.csv").assign(
    epitope=lambda x: x["antibody"].map(antibody_to_epitope),
    escape_fraction=lambda x: (
        x["escape_fraction"].clip(lower=escape_frac_floor, upper=escape_frac_ceil)
    ),
    escape_unadjusted=lambda x: 6.9 - numpy.log(1 / x["escape_fraction"] - 1),
    escape=lambda x: x["escape_unadjusted"].where(x["escape_unadjusted"] > 1.2, 0),
)[["epitope", "mutation", "escape"]]

mut_escape_df.to_csv("RBD_mut_escape_df.csv", index=False, float_format="%.4g")
