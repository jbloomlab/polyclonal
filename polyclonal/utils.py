"""
===========
utils
===========

Miscellaneous utility functions.

"""


import re

import matplotlib.colors


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


def color_gradient_hex(start, end, n):
    """Get a list of colors linearly spanning a range.

    Parameters
    -----------
    start : str
        Starting color.
    end : str
        Ending color.
    n : int
        Number of colors in list.

    Returns
    -------
    list
        List of hex codes for colors spanning `start` to `end`.

    Example
    -------
    >>> color_gradient_hex('white', 'red', n=5)
    ['#ffffff', '#ffbfbf', '#ff8080', '#ff4040', '#ff0000']

    """
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    name='_',
                    colors=[start, end],
                    N=n)
    return [matplotlib.colors.rgb2hex(tup) for tup in cmap(list(range(0, n)))]


if __name__ == '__main__':
    import doctest
    doctest.testmod()
