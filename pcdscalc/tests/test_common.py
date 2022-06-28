import logging

import numpy as np
import pytest

from pcdscalc import common

logger = logging.getLogger(__name__)


@pytest.mark.parametrize('angle, expected', [
                         pytest.param(10, 0.984807753012208),
                         pytest.param(180, -1.0),
                         pytest.param(360, 1.0),
                         pytest.param(45, 0.7071067811865476)
                         ])
def test_cosd(angle, expected):
    res = common.cosd(angle)
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
    res = common.sind(angle)
    logger.debug('Expected: %s Received: %s', expected, res)
    assert np.isclose(expected, res)


@pytest.mark.parametrize('angle, expected', [
                         pytest.param(45, 1),
                         pytest.param(60, np.sqrt(3)),
                         pytest.param(30, 1/np.sqrt(3))
                         ])
def test_tand(angle, expected):
    res = common.tand(angle)
    logger.debug('Expected: %s Received: %s', expected, res)
    assert np.isclose(expected, res)


@pytest.mark.parametrize('x, expected', [
                         pytest.param(-1, -90),
                         pytest.param(1, 90),
                         pytest.param(0.2, 11.536959032815489)
                         ])
def test_asind(x, expected):
    res = common.asind(x)
    logger.debug('Expected: %s Received: %s', expected, res)
    assert np.isclose(expected, res)


@pytest.mark.parametrize('x, expected', [
                         pytest.param(-1, 180),
                         pytest.param(1, 0),
                         pytest.param(0, 90)
                         ])
def test_acosd(x, expected):
    res = common.acosd(x)
    logger.debug('Expected: %s Received: %s', expected, res)
    assert np.isclose(expected, res)


@pytest.mark.parametrize('x, expected', [
                         pytest.param(9, 83.66),
                         pytest.param(1, 45),
                         pytest.param(0, 0)
                         ])
def test_atand(x, expected):
    res = common.atand(x)
    logger.debug('Expected: %s Received: %s', expected, res)
    assert np.isclose(expected, res)


@pytest.mark.parametrize('energy, expected', [
                         pytest.param(2e3, 6.1992e-10),
                         pytest.param(10e3, 1.23984e-10),
                         pytest.param(9e3, 1.3776e-10),
                         pytest.param(8e3, 1.5498e-10)
                         ])
def test_energy_to_wavelength(energy, expected):
    # tested agains the old code
    res = common.energy_to_wavelength(energy)
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
    res = common.wavelength_to_energy(wavelength)
    logger.debug('Expected: %s Received: %s', expected, res)
    assert np.isclose(expected, res)
