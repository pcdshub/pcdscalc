from pcdscalc import be_lens_calcs
import numpy as np
import logging


logger = logging.getLogger(__name__)

sample_lens_set = [2, 200e-6, 4, 500e-6]
sample_energy = 8


def test_get_delta():
    logger.debug('test_get_delta')
    expected = 5.326454632470501e-06
    delta = be_lens_calcs.get_delta(sample_energy)
    logger.debug('Expected: %s, Received %s', expected, delta)
    assert np.isclose(delta, expected)


def test_calc_focal_length_for_single_lens():
    logger.info('test_calc_focal_length_for_single_lens')
    expected = 9.387107081546501
    focal_length = be_lens_calcs.calc_focal_length_for_single_lens(
        sample_energy, .0001)
    logger.debug('Expected: %s, Received: %s',
                 expected, focal_length)
    assert np.isclose(focal_length, expected)


def test_calc_focal_length():
    logger.info('test_calc_focal_length')
    expected = 5.2150594897480556
    fl = be_lens_calcs.calc_focal_length(sample_energy, sample_lens_set)
    logger.debug('Expected: %s, Received: %s', expected, fl)
    assert np.isclose(fl, expected)


def test_calc_beam_fwhm():
    logger.info('test_calc_beam_fwhm')
    expected = 0.00011649743222659306
    fwhm = be_lens_calcs.calc_beam_fwhm(8, sample_lens_set, distance=4,
                                        fwhm_unfocused=500e-6)
    logger.debug('Expected: %s, Received: %s', expected, fwhm)
    assert np.isclose(fwhm, expected)


def test_calc_beam_fwhm_with_source_distance():
    logger.info('test_calc_beam_fwhm_with_source_distance')
    fwhm = be_lens_calcs.calc_beam_fwhm(8, sample_lens_set, distance=4,
                                        fwhm_unfocused=500e-6,
                                        source_distance=10)
    # printing here for curiosity, but need to fix this
    logger.debug('The fwhm with source_distance of 10 is: %s', fwhm)


def test_calc_distance_for_size():
    logger.info('test_calc_beam_fwhm')
    expected = 0.00011649743222659306
    dis = be_lens_calcs.calc_beam_fwhm(8, sample_lens_set, distance=4,
                                       fwhm_unfocused=500e-6)
    logger.debug('Expected: %s, Received: %s', expected, dis)
    assert np.isclose(dis, expected)
