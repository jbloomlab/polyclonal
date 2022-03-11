"""
================================
polyclonal
================================

Package for modeling mutational escape from polyclonal antibodies using
deep mutational scanning experiments.

Importing this package imports the following objects
into the package namespace:

 - :mod:`polyclonal.polyclonal.Polyclonal`

 - :mod:`polyclonal.polyclonal_collection.PolyclonalCollection`

"""

__author__ = "`the Bloom lab <https://research.fhcrc.org/bloom/en.html>`_"
__email__ = "jbloom@fredhutch.org"
__version__ = "0.1"
__url__ = "https://github.com/jbloomlab/polyclonal"

from polyclonal.polyclonal import Polyclonal  # noqa: F401
from polyclonal.polyclonal_collection import PolyclonalCollection  # noqa: F401
