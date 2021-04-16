"""
Calculation utilities related to the LCLS PMPS system.
"""

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
    pass


def get_bitmask(lower, upper, allow, line):
    """
    Given a range of eV values, calculate the appropriate pmps bitmask.

    This saves you the effort of checking up on the eV ranges and
    remembering how the bitmask is assembled.

    Parameters
    ----------
    lower: number
        The value in eV for the lower bound of the range.

    upper: number
        The value in eV for the upper bound of thr range.

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
    pass


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
