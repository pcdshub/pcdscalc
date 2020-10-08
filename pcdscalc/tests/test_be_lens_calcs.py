import logging

import numpy as np
import pytest
from pcdscalc import be_lens_calcs

logger = logging.getLogger(__name__)

sample_lens_set = [2, 200e-6, 4, 500e-6]
sample_energy = 8


@pytest.mark.parametrize('energy_sample, expected', [
    pytest.param(8, 5.326454632470501e-06),
    pytest.param(9, 4.20757010499706e-06),
    pytest.param(1, 0.0003524314661392802),
])
def test_get_delta(energy_sample, expected):
    delta = be_lens_calcs.get_delta(energy_sample)
    logger.debug('Expected: %s, Received %s', expected, delta)
    assert np.isclose(delta, expected)


def test_get_delta_with_0_energy():
    # should give a nan valus since can't devide by 0
    delta = be_lens_calcs.get_delta(0)
    logger.debug('Expected: %s, Received: %s', np.nan, delta)
    assert np.isnan(delta)


@pytest.mark.parametrize('energy_sample, radius, expected', [
    pytest.param(8, 0.003, 281.61321244639504),
    pytest.param(8, 0.0001, 9.387107081546501),
    pytest.param(9, 0.003, 356.5002988823755),
    pytest.param(1, 0.0001, 0.1418715546251483),
])
def test_calc_focal_length_for_single_lens(energy_sample, radius, expected):
    focal_length = be_lens_calcs.calc_focal_length_for_single_lens(
        energy_sample, radius)
    logger.debug('Expected: %s, Received: %s', expected, focal_length)
    assert np.isclose(focal_length, expected)


@pytest.mark.parametrize('energy_sample, lens_set, expected', [
    pytest.param(8, [1, 0.02, 5, 0.004, 2, 1.23, 1, 0.02], 69.45047645294554),
    pytest.param(8, [2, 200e-6, 4, 500e-6], 5.2150594897480556),
    pytest.param(9, [1, 0.02, 5, 0.004, 2, 1.23, 1, 0.02], 87.91887070181892),
    pytest.param(1, [1, 0.02, 5, 0.004, 2, 1.23, 1, 0.02], 1.0496361635424505),
])
def test_calc_focal_length(energy_sample, lens_set, expected):
    fl = be_lens_calcs.calc_focal_length(energy_sample, lens_set)
    logger.debug('Expected: %s, Received: %s', expected, fl)
    assert np.isclose(fl, expected)


@pytest.mark.parametrize('energy_sample, lens_set, dist, fwhm_unf, expect', [
    pytest.param(8, [1, 0.02, 5, 0.004, 2, 1.23, 1, 0.02],
                 4, 800e-6, 0.000753947185823854),
    pytest.param(8, [2, 200e-6, 4, 500e-6],
                 4, 800e-6, 0.00018639295509465447),
    pytest.param(8, [2, 200e-6, 4, 500e-6],
                 3, 500e-6, 0.00021237263787327295),
    pytest.param(8, [1, 0.02, 5, 0.004, 2, 1.23, 1, 0.02],
                 4, 500e-6, 0.0004712974533787206),
    pytest.param(1, [1, 0.02, 5, 0.004, 2, 1.23, 1, 0.02],
                 3, 500e-6, 0.0009290673198171924),
])
def test_calc_beam_fwhm(energy_sample, lens_set, dist, fwhm_unf, expect):
    fwhm = be_lens_calcs.calc_beam_fwhm(energy=energy_sample,
                                        lens_set=lens_set,
                                        distance=dist,
                                        fwhm_unfocused=fwhm_unf)
    logger.debug('Expected: %s, Received: %s', expect, fwhm)
    assert np.isclose(fwhm, expect)


# TODO: this was tested by adding the souce_distance implementation in
# the old code..... so it is kind of cheating....
@pytest.mark.parametrize('energy_sample, lens_set, dist ,'
                         'fwhm_unf, source_dist, expected', [
                             pytest.param(8, [2, 200e-6, 4, 500e-6],
                                          4, 500e-6, 10, 0.000316498748238284),
                         ])
def test_calc_beam_fwhm_with_source_distance(energy_sample, lens_set, dist,
                                             fwhm_unf, source_dist, expected):
    fwhm = be_lens_calcs.calc_beam_fwhm(energy=energy_sample,
                                        lens_set=lens_set, distance=dist,
                                        fwhm_unfocused=fwhm_unf,
                                        source_distance=source_dist)
    # printing here for curiosity, but need to fix this
    logger.debug('The fwhm with source_distance of 10 is: %s', fwhm)
    assert np.isclose(fwhm, expected)


@pytest.mark.parametrize('energy, lens_set, fwhm_unf, size_fwhm, expect', [
    pytest.param(8, [1, 0.02, 5, 0.004, 2, 1.23, 1, 0.02],
                 800e-6, 0.000753947185823854, [4.0, 134.90095291]),
    pytest.param(8, [2, 200e-6, 4, 500e-6],
                 800e-6, 0.00018639295509465447, [4.0, 6.43011898]),
    pytest.param(8, [2, 200e-6, 4, 500e-6],
                 500e-6, 0.00021237263787327295, [3.0, 7.43011898]),
    pytest.param(8, [1, 0.02, 5, 0.004, 2, 1.23, 1, 0.02],
                 500e-6, 0.0004712974533787206, [4.0, 134.90095291]),
    pytest.param(1, [1, 0.02, 5, 0.004, 2, 1.23, 1, 0.02],
                 500e-6, 0.0009290673198171924, [-0.90072767, 3.0]),
])
def test_calc_distance_for_size(energy, lens_set, fwhm_unf, size_fwhm, expect):
    dis = be_lens_calcs.calc_distance_for_size(size_fwhm=size_fwhm,
                                               lens_set=lens_set,
                                               energy=energy,
                                               fwhm_unfocused=fwhm_unf)
    logger.debug('Expected: %s, Received: %s', expect, dis)
    assert np.isclose(dis, expect).all()
