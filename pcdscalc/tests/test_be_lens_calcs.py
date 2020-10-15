import logging
import os
from unittest.mock import patch

import numpy as np
import pytest
from pcdscalc import be_lens_calcs

logger = logging.getLogger(__name__)

PATH = os.path.dirname(__file__) + '/test_lens_sets/lens_set.npy'
BAD_PATH = '../lens_set'

SETS_SAMPLE = [(3, 0.0001, 1, 0.0002),
               (1, 0.0001, 1, 0.0003, 1, 0.0005),
               (2, 0.0001, 1, 0.0005)]


def test_configure_lens_set_bad_path():
    res = be_lens_calcs.configure_lens_set_file(BAD_PATH)
    assert res is None
    assert be_lens_calcs.LENS_SET_FILE is None


def test_configure_lens_set_file():
    res = be_lens_calcs.configure_lens_set_file(PATH)
    assert res == os.path.abspath(PATH)
    assert be_lens_calcs.LENS_SET_FILE == res


def test_set_lens_set_to_file():
    be_lens_calcs.set_lens_set_to_file(SETS_SAMPLE, PATH, make_backup=False)
    assert np.load(PATH, allow_pickle=True).all(
    ) == np.array(SETS_SAMPLE, dtype=object).all()


def test_get_lens_set():
    first_set = (3, 0.0001, 1, 0.0002)
    lens_set = be_lens_calcs.get_lens_set(1, PATH)
    logger.debug(f'result: {lens_set}')
    assert lens_set == first_set


def test_get_lens_set_no_file():
    # should just return because no file provided the LENS_SET_FILE is not
    # configured
    be_lens_calcs.LENS_SET_FILE = None
    res = be_lens_calcs.get_lens_set(1)
    assert res is None


@pytest.mark.parametrize('energy_sample, expected', [
    pytest.param(8, 0.004810113254120656),
    pytest.param(9, 0.006507409652366966),
    pytest.param(1, 8.946976460379717e-06),
    pytest.param(30, 0.030160746908555823),
    pytest.param(10, 0.008360110829541422),
])
def test_get_att_length(energy_sample, expected):
    # old values (with old get_att_lenght)
    # 0.005949593420831057
    # 0.00874765768614978
    # 9.030173195532919e-06
    # 0.4711761144434822
    # 0.01235515814053393
    att_len = be_lens_calcs.get_att_len(energy_sample, material='Be')
    logger.debug('At energy: %s, Expected: %s,  Received %s',
                 energy_sample, expected, att_len)
    assert np.isclose(expected, att_len)


def test_get_att_len_with_bad_material():
    # should raise an exception, provided unknown element string
    with pytest.raises(ValueError):
        be_lens_calcs.get_att_len(8, 'BEES', 32)


@pytest.mark.parametrize('energy_sample, expected', [
    pytest.param(8, 5.330471261281744e-06),
    pytest.param(9, 4.21084077002856e-06),
    pytest.param(1, 0.0003515107795702183),
    pytest.param(0.1, 0.005621922785393444),
    pytest.param(20, 8.521078856544226e-07),
])
def test_get_delta(energy_sample, expected):
    # old values (using old get_data):
    # 5.326454632470501e-06
    # 4.20757010499706e-06
    # 0.0003524314661392802
    # 0.003807346693343705
    # 8.513802953746819e-07
    delta = be_lens_calcs.get_delta(energy_sample, material='Be')
    logger.debug('At energy: %s, Expected: %s, Received %s',
                 energy_sample, expected, delta)
    assert np.isclose(expected, delta)


def test_get_delta_with_0_energy():
    # should give an exception since can't devide by 0
    with pytest.raises(ZeroDivisionError):
        be_lens_calcs.get_delta(0)


@pytest.mark.parametrize('energy_sample, radius, expected', [
    pytest.param(8, 0.003, 281.40101061895905),
    pytest.param(8, 0.0001, 9.380033687298635),
    pytest.param(9, 0.003, 356.2233962102125),
    pytest.param(1, 0.0001, 0.14224314844948285),
])
def test_calc_focal_length_for_single_lens(energy_sample, radius, expected):
    # old values (with old get_delta):
    # 281.61321244639504
    # 9.387107081546501
    # 356.5002988823755
    # 0.1418715546251483
    focal_length = be_lens_calcs.calc_focal_length_for_single_lens(
        energy_sample, radius)
    logger.debug('Expected: %s, Received: %s', expected, focal_length)
    assert np.isclose(focal_length, expected)


@pytest.mark.parametrize('energy_sample, lens_set, expected', [
    pytest.param(8, [1, 0.02, 5, 0.004, 2, 1.23, 1, 0.02], 69.39814397219442),
    pytest.param(8, [2, 200e-6, 4, 500e-6], 5.21112982627702),
    pytest.param(9, [1, 0.02, 5, 0.004, 2, 1.23, 1, 0.02], 87.85058192251857),
    pytest.param(1, [1, 0.02, 5, 0.004, 2, 1.23, 1, 0.02], 1.0523853990548204),
])
def test_calc_focal_length(energy_sample, lens_set, expected):
    # old values (with old get_delta):
    # 69.45047645294554
    # 5.2150594897480556
    # 87.91887070181892
    # 1.0496361635424505
    fl = be_lens_calcs.calc_focal_length(energy_sample, lens_set)
    logger.debug('Expected: %s, Received: %s', expected, fl)
    assert np.isclose(fl, expected)


def test_calc_focal_length_with_file_lens_set():
    # expected_used_set = (3, 0.0001, 1, 0.0002)
    with patch('pcdscalc.be_lens_calcs.get_lens_set',
               return_value=[3, 0.0001, 1, 0.0002]):
        expected = be_lens_calcs.calc_focal_length(8, [3, 0.0001, 1, 0.0002])
        received = be_lens_calcs.calc_focal_length(8, 1)
        logger.debug('Expected: %s, Received: %s', expected, received)
        assert received == expected


def test_calc_focal_length_with_no_file():
    # make sure the LENS_SET_FILE is none
    # should get an Exception
    be_lens_calcs.LENS_SET_FILE = None
    with pytest.raises(Exception):
        be_lens_calcs.calc_focal_length(8, 1)


@pytest.mark.parametrize('energy_sample, lens_set, dist, fwhm_unf, expect', [
    pytest.param(8, [1, 0.02, 5, 0.004, 2, 1.23, 1, 0.02],
                 4, 800e-6, 0.0007539125970041841),
    pytest.param(8, [2, 200e-6, 4, 500e-6],
                 4, 800e-6, 0.00018593024432864825),
    pytest.param(8, [2, 200e-6, 4, 500e-6],
                 3, 500e-6, 0.00021215574911270682),
    pytest.param(8, [1, 0.02, 5, 0.004, 2, 1.23, 1, 0.02],
                 4, 500e-6, 0.0004712763789963477),
    pytest.param(1, [1, 0.02, 5, 0.004, 2, 1.23, 1, 0.02],
                 3, 500e-6, 0.0009253340604551429),
])
def test_calc_beam_fwhm(energy_sample, lens_set, dist, fwhm_unf, expect):
    # old values (with old get_delta):
    # 0.000753947185823854
    # 0.00018639295509465447
    # 0.00021237263787327295
    # 0.0004712974533787206
    # 0.0009290673198171924
    fwhm = be_lens_calcs.calc_beam_fwhm(energy=energy_sample,
                                        lens_set=lens_set,
                                        distance=dist,
                                        fwhm_unfocused=fwhm_unf)
    logger.debug('Expected: %s, Received: %s', expect, fwhm)
    assert np.isclose(fwhm, expect)


@pytest.mark.parametrize('energy_sample, lens_set, dist ,'
                         'fwhm_unf, source_dist, expected', [
                             pytest.param(8, [2, 200e-6, 4, 500e-6],
                                          4, 500e-6, 10,
                                          0.0003162095431898309),
                         ])
def test_calc_beam_fwhm_with_source_distance(energy_sample, lens_set, dist,
                                             fwhm_unf, source_dist, expected):
    # old values (with old get_delta):
    # 0.000316498748238284
    fwhm = be_lens_calcs.calc_beam_fwhm(energy=energy_sample,
                                        lens_set=lens_set, distance=dist,
                                        fwhm_unfocused=fwhm_unf,
                                        source_distance=source_dist)
    logger.debug('Expected: %s, Received: %s', expected, fwhm)
    assert np.isclose(fwhm, expected)


@pytest.mark.parametrize('energy, lens_set, fwhm_unf, size_fwhm, expect', [
    pytest.param(8, [1, 0.02, 5, 0.004, 2, 1.23, 1, 0.02],
                 800e-6, 0.0007539124064936633, [4.00001653, 134.79627142]),
    pytest.param(8, [2, 200e-6, 4, 500e-6],
                 800e-6, 0.00018593023997294836, [4.00000003, 6.42225962]),
    pytest.param(8, [2, 200e-6, 4, 500e-6],
                 500e-6, 0.0002121557393404887, [3.0000001, 7.42225955]),
    pytest.param(8, [1, 0.02, 5, 0.004, 2, 1.23, 1, 0.02],
                 500e-6, 0.00047127559879883534, [4.00010831, 134.79617963]),
    pytest.param(1, [1, 0.02, 5, 0.004, 2, 1.23, 1, 0.02],
                 500e-6, 0.0009253340546070372, [-0.89522919, 2.99999999]),
])
def test_calc_distance_for_size(energy, lens_set, fwhm_unf, size_fwhm, expect):
    # old values (with old get_delta):
    # [4.00301939, 134.89793351]
    # [4.00301637, 6.42710261]
    # [3.00226229, 7.42785669]
    # [4.00303623, 134.89791667]
    # [-0.89289053,  2.99216285]
    dis = be_lens_calcs.calc_distance_for_size(size_fwhm=size_fwhm,
                                               lens_set=lens_set,
                                               energy=energy,
                                               fwhm_unfocused=fwhm_unf)
    logger.debug('Expected: %s, Received: %s', expect, dis)
    assert np.isclose(dis, expect).all()


@pytest.mark.parametrize('energy_sample, distance, expected', [
    pytest.param(8, 4, 4.2643770090253954e-05),
    pytest.param(8, 3, 3.198282756769046e-05),
    pytest.param(9, 4, 3.368672616022848e-05),
    pytest.param(10, 2.5, 1.7051278100008672e-05),
])
def test_find_radius(energy_sample, distance, expected):
    # Old values, with old get_delta
    # 4.261163705976401e-05
    # 3.1958727794823005e-05
    # 3.366056083997648e-05
    # 1.7037713265222187e-05
    radius = be_lens_calcs.find_radius(energy=energy_sample, distance=distance)
    logger.debug('Expected: %s, Received: %s', expected, radius)
    assert np.isclose(expected, radius)


@pytest.mark.parametrize('radius, expected', [
    pytest.param(1.0e-3, 0.0009848857801796104),
    pytest.param(2.0e-3, 0.001392838827718412),
    pytest.param(3.0e-3, 0.0017058722109231982),
])
def test_calc_lens_aperture_radius(radius, expected):
    aperture_radius = be_lens_calcs.calc_lens_aperture_radius(radius)
    logger.debug('Expected: %s, Received: %s', expected,
                 aperture_radius)
    assert np.isclose(expected, aperture_radius)


@pytest.mark.parametrize('energy_sample, radius, expected', [
    pytest.param(8, 1.0e-3, 0.909202465413282),
    pytest.param(8, 2.0e-3, 0.9634630968766913),
    pytest.param(9, 3.0e-3, 0.9806842723309849),
])
def test_calc_trans_for_single_lens(energy_sample, radius, expected):
    # old values (with old get_att_len and old get_delta):
    # 0.9192789600686936
    # 0.9700496829425548
    # 0.9855626484449255
    trans = be_lens_calcs.calc_trans_for_single_lens(energy=energy_sample,
                                                     radius=radius,
                                                     fwhm_unfocused=900e-6)
    logger.debug('Expected: %s, Received: %s', expected, trans)
    assert np.isclose(expected, trans)


@pytest.mark.parametrize('energy_sample, lens_set, expected', [
    pytest.param(8, [2, 100e-6, 4, 200e-6], 0.1909312006707346),
    pytest.param(8, [2, 200e-6, 4, 500e-6], 0.3455637290620128),
    pytest.param(9, [2, 200e-6, 4, 500e-6], 0.3760514142257077),
])
def test_calc_trans_lens_set(energy_sample, lens_set, expected):
    # old values (with old get_att_len and old get_delta):
    # 0.20497965794420414
    # 0.3675986395544651
    # 0.4005461885189993
    trans = be_lens_calcs.calc_trans_lens_set(
        energy_sample, lens_set, fwhm_unfocused=900e-6)
    logger.debug('Expected: %s Received: %s', expected, trans)
    assert np.isclose(expected, trans)


def test_calc_trans_lens_set_with_file_len_set():
    # expected_used_set = (3, 0.0001, 1, 0.0002)
    with patch('pcdscalc.be_lens_calcs.get_lens_set',
               return_value=[3, 0.0001, 1, 0.0002]):
        expected = be_lens_calcs.calc_trans_lens_set(8, [3, 0.0001, 1, 0.0002])

        received = be_lens_calcs.calc_trans_lens_set(8, 1)
        logger.debug('Expected: %s, Received: %s', expected, received)
        assert received == expected


def test_calc_trans_lens_set_with_no_file():
    # make sure the LENS_SET_FILE is none
    # should get an Exception
    be_lens_calcs.LENS_SET_FILE = None
    with pytest.raises(Exception):
        be_lens_calcs.calc_trans_lens_set(8, 1)


@pytest.mark.parametrize('lens_set, distance, expected', [
                         pytest.param([2, 200e-6, 4, 500e-6], 4,
                                      7.0100555419921875),
                         pytest.param([2, 200e-6, 4, 500e-6], 2,
                                      4.9597930908203125),
                         pytest.param([2, 200e-6, 4, 500e-6], 7,
                                      9.270801544189453)
                         ])
def test_find_energy(lens_set, distance, expected):
    # NOTE: the logs from calc_beam_fwhm differ here from the
    # old ones... but the final results seem to be the same???
    # old values (with old get_att_len and old get_delta):
    # 7.007423400878906
    # 4.958213806152344
    # 9.267204284667969
    energy = be_lens_calcs.find_energy(lens_set, distance)
    logger.debug('Expected: %s Received: %s', expected, energy)
    assert np.isclose(expected, energy)


@pytest.mark.parametrize('energy, lens_set, spot_size_fwhm, expected', [
                         pytest.param(8, [2, 200e-6, 4, 500e-6], 4.0e-3,
                                      (-20.844519143534555,
                                      31.266778796088595))
                         ])
def test_find_z_pos(energy, lens_set, spot_size_fwhm, expected):
    # old values (with old get_att_len and old get_delta):
    # (-20.86023779837472, 31.290356777870826)
    z_position = be_lens_calcs.find_z_pos(energy, lens_set, spot_size_fwhm,
                                          fwhm_unfocused=800e-6)
    logger.debug('Expected: %s, Received: %s', expected, z_position)
    assert np.isclose(expected, z_position).all()


def test_calc_lens_set():
    expected_sets = np.array([[0, 0, 0, 0, 1],
                             [0, 0, 0, 1, 0],
                             [0, 0, 1, 0, 0],
                             [0, 1, 0, 0, 0],
                             [1, 0, 0, 0, 0]])

    expected_effrads = np.array([0.001, 0.0005, 0.0003, 0.0002, 0.0001])
    expected_sizes = np.array([0.00047886, 0.00045743, 0.000429,
                              0.00039348, 0.00028694])
    expected_foclens = np.array([93.87107082, 46.93553541, 28.16132124,
                                18.77421416,  9.38710708])

    res = be_lens_calcs.calc_lens_set(energy=8, size_fwhm=2, distance=4,
                                      n_max=1, max_each=2,
                                      lens_radii=[100e-6, 200e-6, 300e-6,
                                                  500e-6, 1000e-6],
                                      fwhm_unfocused=0.0005, eff_rad0=None)
    sets = res[0]
    effrads = res[1]
    sizes = res[2]
    foclens = res[3]
    assert sets.all() == expected_sets.all()
    assert effrads.all() == expected_effrads.all()
    assert sizes.all() == expected_sizes.all()
    assert foclens.all() == expected_foclens.all()
