import logging

import pytest

from pcdscalc.pmps import (LFE, KFE, select_bitmask_boundaries,
                           get_bitmask, check_bitmask)


logger = logging.getLogger(__name__)

# 32 bits, using numbers from 1 to 33
test_boundaries = list(range(1,33))

# Define some utility bitmasks
allow_none = 0
allow_all = 2**32-1
bm1 = 0b1111_0000_0000_0000_0000_0000_0000_1111
bm2 = 0b0000_0000_0000_1111_1111_0000_0000_0000
bm3 = 0b0000_0000_0000_0001_0000_0000_0000_0000
bm4 = 0b1111_1111_1111_1110_1111_1111_1111_1111


@pytest.mark.parametrize(
    "test_input,expected",
    [('k', KFE), ('kfe', KFE), ('sxr', KFE),
     ('L', LFE), ('LFE', LFE), ('HXR', LFE)]
    )
def test_select_bounds(test_input, expected):
    logger.debug(f'test_select_bounds({test_input}, {expected})')
    assert select_bitmask_boundaries(test_input) is expected


@pytest.mark.parametrize(
    "lower,upper,allow,expected",
    [(0, 100, True, allow_all),
     (0, 100, False, allow_none),
     (0, 15.5, True,     0b0000_0000_0000_0000_0111_1111_1111_1111),
     (15.5, 100, True,   0b1111_1111_1111_1111_0000_0000_0000_0000),
     (14.5, 21.5, True,  0b0000_0000_0001_1111_1000_0000_0000_0000),
     (14.5, 21.5, False, 0b1111_1111_1100_0000_0011_1111_1111_1111),
     (15, 20, True,      0b0000_0000_0000_1111_1000_0000_0000_0000),
    ])
def test_get_bitmask(lower, upper, allow, expected):
    logger.debug(f'test_get_bitmask({lower}, {upper}, {allow}, {expected})')
    bitmask = get_bitmask(lower, upper, allow, 'tst', bounds=test_boundaries)
    assert bitmask == expected


@pytest.mark.parametrize(
    "energy,bitmask,expected",
    [(0, bm1, True), (16, bm1, False), (30, bm1, True), (40, bm1, False),
     (0, bm2, False), (16, bm2, True), (30, bm2, False), (-1, bm2, False),
     (7, bm3, False), (16, bm3, True), (17, bm3, True), (30, bm3, False),
     (0, bm4, True), (16, bm4, False), (17, bm4, False), (30, bm4, True),
    ])
def test_check_bitmask(energy, bitmask, expected):
    logger.debug(f'test_check_bitmask({energy}, {bitmask}, {expected}')
    ok = check_bitmask(energy, bitmask, 'tst', bounds=test_boundaries)
    assert ok == expected
