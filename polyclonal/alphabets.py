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


def biochem_order_aas(alphabet):
    """Put amino-acids in "biochemical order" so ones with similar properties are nearby.

    Parameters
    ----------
    alphabet : array-like
        Amino-acid characters, can include stop (``*``) and gap (``-``).

    Returns
    -------
    tuple
        Contains the alphabet in biochemical order

    Example
    -------

    >>> biochem_order_aas(AAS_WITHSTOP_WITHGAP)
    ... # doctest: +NORMALIZE_WHITESPACE
    ('R', 'K', 'H', 'D', 'E', 'Q', 'N', 'S', 'T', 'Y', 'W', 'F', 'A', 'I', 'L', 'M',
    'V', 'G', 'P', 'C', '-', '*')

    """
    sort_order = {
        a: i
        for i, a in enumerate(
            [
                "R",
                "K",
                "H",
                "D",
                "E",
                "Q",
                "N",
                "S",
                "T",
                "Y",
                "W",
                "F",
                "A",
                "I",
                "L",
                "M",
                "V",
                "G",
                "P",
                "C",
                "-",
                "*",
            ]
        )
    }

    if len(alphabet) != len(set(alphabet)):
        raise ValueError(f"Duplicate letters in {alphabet=}")

    if not set(alphabet).issubset(sort_order):
        raise ValueError(f"Invalid letters in {alphabet=}")

    return tuple(sorted(alphabet, key=lambda a: sort_order[a]))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
