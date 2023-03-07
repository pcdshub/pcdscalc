"""
Calculation utilities related to the LCLS PMPS system.

Here you will find functions related to eV bitmasks.
"""
import math
from typing import Optional

# Source for these values:
# lcls-twincat-pmps/PMPS/GVLs/PMPS_GVL.TcGVL
LFE = [
    1.0E3,
    1.7E3,
    2.1E3,
    2.5E3,
    3.8E3,
    4.0E3,
    5.0E3,
    7.0E3,
    7.5E3,
    7.7E3,
    8.9E3,
    10.0E3,
    11.1E3,
    12.0E3,
    13.0E3,
    13.5E3,
    14.0E3,
    16.9E3,
    18.0E3,
    20.0E3,
    22.0E3,
    24.0E3,
    25.0E3,
    25.5E3,
    26.0E3,
    27.0E3,
    28.0E3,
    28.5E3,
    29.0E3,
    30.0E3,
    60.0E3,
    90.0E3,
]

KFE = [
    100,
    250,
    270,
    350,
    400,
    450,
    480,
    530,
    680,
    730,
    850,
    1.10E3,
    1.15E3,
    1.25E3,
    1.45E3,
    1.50E3,
    1.55E3,
    1.65E3,
    1.70E3,
    1.75E3,
    1.82E3,
    1.85E3,
    2.00E3,
    2.20E3,
    2.50E3,
    2.80E3,
    3.00E3,
    3.15E3,
    3.50E3,
    4.00E3,
    5.30E3,
    7.00E3
]


def select_bitmask_boundaries(line: str) -> list[float]:
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


def get_bitmask(
    lower: float,
    upper: float,
    allow: bool,
    line: str,
    bounds: Optional[list[float]] = None,
) -> int:
    """
    Given a range of eV values, calculate the appropriate pmps bitmask.

    This saves you the effort of checking up on the eV ranges and
    remembering how the bitmask is assembled.

    If the lower or upper fall within a range (not on the border), then that
    range is always considered unsafe.

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

    bounds: list of numbers, optional
        Custom boundaries to use instead of the default soft-xray
        or hard-xray lines. Useful for testing.

    Returns
    -------
    bitmask: int
    """
    # Help the user if they accidently pass a negative
    if lower < 0 or upper < 0:
        raise ValueError('get_bitmask is only valid for positive inputs')

    # Be lenient on input args
    if upper < lower:
        lower, upper = upper, lower

    if allow:
        bounds = bounds or select_bitmask_boundaries(line)
        bitmask = 0

        prev = 0
        for bit, ev in enumerate(bounds):
            if lower <= prev and upper >= ev:
                bitmask += 2**bit
            prev = ev
        return bitmask

    # An exclusion range is just two inclusion ranges
    else:
        lower_range = get_bitmask(0, lower, True, line, bounds=bounds)
        upper_range = get_bitmask(upper, math.inf, True, line, bounds=bounds)
        return lower_range | upper_range


def check_bitmask(
    energy: float,
    bitmask: int,
    line: str,
    bounds: Optional[list[float]] = None,
) -> bool:
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

    bounds: list of numbers, optional
        Custom boundaries to use instead of the default soft-xray
        or hard-xray lines. Useful for testing.

    Returns
    -------
    energy_allowed: bool
        True if the energy is allowed.
    """
    bounds = bounds or select_bitmask_boundaries(line)

    # Boundary energy exists in two ranges, so we can get two answers
    answers = []
    prev = 0
    for bit, ev in enumerate(bounds):
        if prev <= energy <= ev:
            ok = bool((bitmask >> bit) % 2)
            answers.append(ok)
        prev = ev

    if not answers:
        # We get here if energy is negative or too large
        return False
    else:
        return all(answers)


def check_actual_range(
    lower: float,
    upper: float,
    allow: bool,
    line: str,
    bounds: Optional[list[float]] = None,
) -> tuple[float, float]:
    """
    Returns the actual effective range given bitmask precision.

    Because of the granularity of the bitmask, most range specifications
    exclude more energy values than requested.

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

    bounds: list of numbers, optional
        Custom boundaries to use instead of the default soft-xray
        or hard-xray lines. Useful for testing.

    Returns
    -------
    ranges: tuple
        A (lower, upper) pair that represents a range of allowed
        (or forbidden) energy values. The endpoints of the range
        are considered unsafe.
    """
    bitmask = get_bitmask(lower, upper, allow, line, bounds=bounds)
    if not allow:
        bitmask = ~bitmask
    lowest = math.inf
    highest = -math.inf
    updated_range = False

    prev = 0
    for bit, ev in enumerate(bounds):
        ok = bool((bitmask >> bit) % 2)
        if ok:
            lowest = min(lowest, prev)
            highest = max(highest, ev)
            updated_range = True
        prev = ev
    if updated_range:
        return (lowest, highest)
    else:
        # The range is empty: return an empty range instead of inf inf.
        return (lower, lower)


def describe_bitmask(
    bitmask: int,
    line: str,
    bounds: Optional[list[float]] = None,
) -> None:
    """
    Print a text description of a bitmask.

    This will describe what the bitmask means.

    Parameters
    ----------
    bitmask : int
        The bits to describe. Typically an output of `get_bitmask`.

    line : str
        String representation of which line's bitmask to use.
        If the string begins with "l" or "h" (lfe, hxr), we'll
        use the hard-xray bitmask.
        If the string begins with "k" or "s" (kfe, sxr), we'll
        use the soft-xray bitmask.

    bounds : list of numbers, optional
        Custom boundaries to use instead of the default soft-xray
        or hard-xray lines. Useful for testing.
    """
    lines = get_bitmask_desc(bitmask=bitmask, line=line, bounds=bounds)
    print('\n'.join(lines))


def get_bitmask_desc(
    bitmask: int,
    line: str,
    bounds: Optional[list[float]] = None,
) -> list[str]:
    """
    Return a text description of a bitmask.

    This will describe what the bitmask means.

    Parameters
    ----------
    bitmask : int
        The bits to describe. Typically an output of `get_bitmask`.

    line : str
        String representation of which line's bitmask to use.
        If the string begins with "l" or "h" (lfe, hxr), we'll
        use the hard-xray bitmask.
        If the string begins with "k" or "s" (kfe, sxr), we'll
        use the soft-xray bitmask.

    bounds : list of numbers, optional
        Custom boundaries to use instead of the default soft-xray
        or hard-xray lines. Useful for testing.

    Returns
    -------
    desc : list of str
        Description lines that describe what the bitmask means.
    """
    bounds = bounds or select_bitmask_boundaries(line)
    lines = []

    width = 0
    for bound in bounds:
        width = max(width, len(str(bound)))

    prev = 0
    for bit, ev in enumerate(bounds):
        val = (bitmask >> bit) % 2
        count = bit + 1
        if val:
            text = 'allowed'
        else:
            text = 'disallowed'
        line = f'Bit {count:2}: {val} ({prev:{width}}, {ev:{width}}) {text}'
        lines.append(line)
        prev = ev
    return lines
