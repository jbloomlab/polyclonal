"""
================================
alphabets
================================

Alphabets for the protein sequences.

"""

import binarymap

AAS = binarymap.binarymap.AAS_NOSTOP
"""tuple: Amino-acid one-letter codes alphabetized, doesn't include stop or gap."""

AAS_WITHSTOP = binarymap.binarymap.AAS_WITHSTOP
"""tuple: Amino-acid one-letter codes alphabetized plus stop as ``*``."""

AAS_WITHGAP = binarymap.binarymap.AAS_WITHGAP
"""tuple: Amino-acid one-letter codes alphabetized plus gap as ``-``."""

AAS_WITHSTOP_WITHGAP = binarymap.binarymap.AAS_WITHSTOP_WITHGAP
"""tuple: Amino-acid one-letter codes plus stop as ``*`` and gap as ``-``."""
