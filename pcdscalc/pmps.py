"""
Calculation utilities related to the LCLS PMPS system.
"""

LFE = [1000 + n * 1000 for n in range(32)]
KFE = [1000 + n * 100 for n in range(32)]


def select_bitmask_boundaries(line):
    """
    Given a line, select the bitmask boundaries to use.

    These are hard-coded for now but in theory this could be loaded from
    a database, an EPICS PV, or some other source.

    Parameters
    ----------
    line: str
        String representation of which line's bitmask to use.
        If the string begins with "l" or "h" (lfe, hxr), we'll
        use the hard-xray bitmask.
        If the string begins with "k" or "s" (kfe, sxr), we'll
        use the soft-xray bitmask.

    Returns
    -------
    boundaries: list of floats
    """
    if line.lower()[0] in ['l', 'h']:
        return LFE
    if line.lower()[0] in ['k', 's']:
        return KFE
    raise ValueError(f'{line} is neither lfe or kfe!.')


def get_bitmask(lower, upper, allow, line):
    """
    Given a range of eV values, calculate the appropriate pmps bitmask.

    This saves you the effort of checking up on the eV ranges and
    remembering how the bitmask is assembled.

    The rules for the bitmasks are:
        - The nth bit of the bitmask represents a range from the n-1th value
          to the nth value
        - If the bit is 1, the range is allowed and is safe.
        - If the bit is 0, the range is not allowed and is not safe.
        - eVs above the highest boundary are never safe
        - negative eVs are never safe

    Some examples:
        - bitmask 00 -> no beam range is safe
        - bitmask 01 -> only the lowest boundary and below are safe
        - bitmask 11 -> only the two lowest boundaries and below are safe
        - bitmask 10 -> only the space between the lowest and second-lowest
                        boundaries are safe.
        - bitmask all 1s -> every eV is OK except for above the highest
                            boundary and except for negative eV

    Parameters
    ----------
    lower: number
        The value in eV for the lower bound of the range.

    upper: number
        The value in eV for the upper bound of the range.

    allow: bool
        True if we want a bitmask that only includes this range,
        False if we want a bitmask that only excludes this range.

    line: str
        String representation of which line's bitmask to use.
        If the string begins with "l" or "h" (lfe, hxr), we'll
        use the hard-xray bitmask.
        If the string begins with "k" or "s" (kfe, sxr), we'll
        use the soft-xray bitmask.

    Returns
    -------
    bitmask: int
    """
    bounds = select_bitmask_boundaries(line)
    bitmask = 0

    prev = 0
    for bit, ev in enumerate(bounds):
        if lower <= prev and upper >= ev:
            bitmask += 2**bit
        prev = ev

    if not allow:
        # TODO this isn't quite right
        # the boundary of the range needs special care, the 0 needs to leak in
        # e.g. 00111000 needs to become 1000011
        return ~bitmask
    return bitmask


def check_bitmask(energy, bitmask, line):
    """
    Given an energy and a bitmask, tell us if our energy is allowed.

    This is the same calculation the PMPS is doing internally to determine
    if it is safe for beam to proceed.

    Parameters
    ----------
    energy: number
        The value in eV for the energy to check.

    bitmask: int
        The bits to check against. Typically an output of `get_bitmask`.

    line: str
        String representation of which line's bitmask to use.
        If the string begins with "l" or "h" (lfe, hxr), we'll
        use the hard-xray bitmask.
        If the string begins with "k" or "s" (kfe, sxr), we'll
        use the soft-xray bitmask.

    Returns
    -------
    energy_allowed: bool
        True if the energy is allowed.
    """
    pass
