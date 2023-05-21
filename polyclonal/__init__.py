"""
================================
polyclonal
================================

Package for modeling mutational escape from polyclonal antibodies using
deep mutational scanning experiments.

Importing this package imports the following objects
into the package namespace:

 - :mod:`~polyclonal.polyclonal.Polyclonal`

 - :mod:`~polyclonal.polyclonal_collection.PolyclonalCollection`

 - :mod:`~polyclonal.polyclonal_collection.PolyclonalAverage`

 - :mod:`~polyclonal.polyclonal_collection.PolyclonalBootstrap`

It also imports the following alphabets:

 - :const:`~polyclonal.alphabets.AAS`

 - :const:`~polyclonal.alphabets.AAS_WITHSTOP`

 - :const:`~polyclonal.alphabets.AAS_WITHGAP`

 - :const:`~polyclonal.alphabets.AAS_WITHSTOP_WITHGAP`

"""

__author__ = "`the Bloom lab <https://research.fhcrc.org/bloom/en.html>`_"
__email__ = "jbloom@fredhutch.org"
__version__ = "6.0"
__url__ = "https://github.com/jbloomlab/polyclonal"

from polyclonal.alphabets import AAS
from polyclonal.alphabets import AAS_WITHGAP
from polyclonal.alphabets import AAS_WITHSTOP
from polyclonal.alphabets import AAS_WITHSTOP_WITHGAP
from polyclonal.polyclonal import Polyclonal
from polyclonal.polyclonal_collection import PolyclonalAverage
from polyclonal.polyclonal_collection import PolyclonalBootstrap
from polyclonal.polyclonal_collection import PolyclonalCollection
