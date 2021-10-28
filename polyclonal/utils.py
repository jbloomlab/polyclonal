"""
===========
utils
===========

Miscellaneous utility functions.

"""


import re


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
