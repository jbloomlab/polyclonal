"""
===========
utils
===========

Miscellaneous utility functions.

"""


import re

from binarymap.binarymap import AAS_NOSTOP


class MutationParser:
    """Parse mutation strings like 'A5G'.

    Parameters
    ----------
    alphabet : array-like
        Valid single-character letters in alphabet.

    Example
    -------
    >>> mutparser = MutationParser(AAS_NOSTOP)
    >>> mutparser.parse_mut('A5G')
    ('A', 5, 'G')

    """

    def __init__(self, alphabet):
        """See main class docstring."""
        chars = []
        for char in alphabet:
            if char.isalpha():
                chars.append(char)
            elif char == '*':
                chars.append(r'\*')
            else:
                raise ValueError(f"invalid alphabet character: {char}")
        chars = '|'.join(chars)
        self._mutation_regex = re.compile(rf"(?P<wt>{chars})"
                                          rf"(?P<site>\d+)"
                                          rf"(?P<mut>{chars})")

    def parse_mut(self, mutation):
        """tuple: `(wildtype, site, mutation)`."""
        m = self._mutation_regex.fullmatch(mutation)
        if not m:
            raise ValueError(f"invalid mutation {mutation}")
        else:
            return (m.group('wt'), int(m.group('site')), m.group('mut'))


def site_level_variants(df,
                        *,
                        original_alphabet=AAS_NOSTOP,
                        wt_char='w',
                        mut_char='m',
                        ):
    """Re-define variants simply in terms of which sites are mutated."""
    raise NotImplementedError


def shift_mut_site(mut_str, shift):
    """Shift site in string of mutations.

    Parameters
    ----------
    mut_str : str
        String of space-delimited amino-acid substitution mutations.
    shift : int
        Amount to shift sites (add this to current site number).

    Returns
    -------
    str
        Mutation string with sites shifted.

    Example
    -------
    >>> shift_mut_site('A1G K7A', 2)
    'A3G K9A'

    """
    if not isinstance(shift, int):
        raise ValueError('shift must be int')
    new_mut_str = []
    for mut in mut_str.split():
        m = re.fullmatch(r'(?P<wt>\S)(?P<site>\d+)(?P<mut>\S)', mut)
        if not m:
            raise ValueError(f"cannot match {mut} in {mut_str}")
        new_site = int(m.group('site')) + shift
        new_mut_str.append(f"{m.group('wt')}{new_site}{m.group('mut')}")
    return ' '.join(new_mut_str)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
