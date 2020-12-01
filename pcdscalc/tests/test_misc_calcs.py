import logging
import numpy as np
import pytest
from pcdscalc import misc_calcs

logger = logging.getLogger(__name__)


@pytest.mark.parametrize('angle, expected', [
                         pytest.param(10, 0.984807753012208),
                         pytest.param(180, -1.0),
                         pytest.param(360, 1.0),
                         pytest.param(45, 0.7071067811865476)
                         ])
def test_cosd(angle, expected):
    res = misc_calcs.cosd(angle)
    logger.debug('Expected: %s Received: %s', expected, res)
    assert np.isclose(expected, res)


@pytest.mark.parametrize('angle, expected', [
                         pytest.param(10, 0.17364817766693033),
                         pytest.param(180, 1.2246467991473532e-16),
                         pytest.param(360, -2.4492935982947064e-16),
                         pytest.param(45, 0.7071067811865476),
                         pytest.param(270, -1.0)
                         ])
def test_sind(angle, expected):
    res = misc_calcs.sind(angle)
    logger.debug('Expected: %s Received: %s', expected, res)
    assert np.isclose(expected, res)


@pytest.mark.parametrize('angle, expected', [
                         pytest.param(-1, -90),
                         pytest.param(1, 90),
                         pytest.param(0.2, 11.536959032815489)
                         ])
def test_asind(angle, expected):
    res = misc_calcs.asind(angle)
    logger.debug('Expected: %s Received: %s', expected, res)
    assert np.isclose(expected, res)


@pytest.mark.parametrize('energy, expected', [
                         pytest.param(100, 100.0),
                         pytest.param(9, 9000.0),
                         pytest.param(99, 99000.0),
                         pytest.param(10, 10000.0)
                         ])
def test_to_ev(energy, expected):
    res = misc_calcs.to_ev(energy)
    logger.debug('Expected: %s Received: %s', expected, res)
    assert np.isclose(expected, res)


@pytest.mark.parametrize('energy, sub, expected', [
                         pytest.param(10, False, 1.23984e-10),
                         pytest.param(10, True, 1.2398399999999998e-07),
                         pytest.param(9, None, 1.3776e-10),
                         pytest.param(100, False,  1.23984e-08)
                         ])
def test_energy_to_wavelength(energy, sub, expected):
    # tested agains the old code
    res = misc_calcs.energy_to_wavelength(energy, sub)
    logger.debug('Expected: %s Received: %s', expected, res)
    assert np.isclose(expected, res)


@pytest.mark.parametrize('energy, material, reflection, expected', [
                         pytest.param(10, 'Si', (1, 1, 1),
                                      (11.402710639982848, 713.4828146545175)),
                         pytest.param(10, 'C', (1, 1, 1),
                                      (17.51878596767417, 427.8469911590626)),
                         pytest.param(9, 'Si', (2, 2, 2),
                                      (26.061879662584946, 233.3438968335023)),
                         pytest.param(9, 'C', (2, 2, 2),
                                      (41.98453277509927, 31.69504158241886))
                         ])
def test_get_geometry(energy, material, reflection, expected):
    # tested agains the old code
    res = misc_calcs.get_geometry(energy, material, reflection)
    logger.debug('Expected: %s Received: %s', expected, res)
    assert np.isclose(expected, res).all()


@pytest.mark.parametrize('material_id, hkl, energy, expected', [
                         pytest.param('Si', (1, 1, 1), 10, 11.402710639982848),
                         pytest.param('Si', (1, 1, 1), 7, 16.405520156903474),
                         pytest.param('C', (1, 1, 1), 10, 17.51878596767417),
                         pytest.param('C', (2, 2, 2), 10, 37.015924623731436),
                         pytest.param('Si', (2, 2, 2), 7, 34.39310890947986),
                         pytest.param('C', (2, 2, 2), None, 37.90277660759642)
                         ])
def test_bragg_angle(material_id, hkl, energy, expected):
    # tested agains the old code
    res = misc_calcs.bragg_angle(material_id, hkl, energy)
    logger.debug('Expected: %s Received: %s', expected, res)
    assert np.isclose(expected, res)


@pytest.mark.parametrize('wavelength, expected', [
                         pytest.param(100, 1.23984e-08),
                         pytest.param(9, 1.3775999999999998e-07),
                         pytest.param(7, 1.7712e-07),
                         pytest.param(0.8, 1.5498e-06)
                         ])
def test_wavelength_to_energy(wavelength, expected):
    # tested agains the old code
    res = misc_calcs.wavelength_to_energy(wavelength)
    logger.debug('Expected: %s Received: %s', expected, res)
    assert np.isclose(expected, res)


@pytest.mark.parametrize('material_id, hkl, expected', [
                         pytest.param('Si', (1, 1, 1), 3.1356011476493755e-10),
                         pytest.param('Si', (2, 2, 2), 1.5678005738246877e-10),
                         pytest.param('C', (1, 1, 1), 2.059408410199395e-10),
                         pytest.param('C', (2, 2, 2), 1.0297042050996975e-10)
                         ])
def test_d_space(material_id, hkl, expected):
    # tested agains the old code
    res = misc_calcs.d_space(material_id, hkl)
    logger.debug('Expected: %s Received: %s', expected, res)
    assert np.isclose(expected, res)
