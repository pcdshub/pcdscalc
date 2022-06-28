import logging

import numpy as np
import pytest

from pcdscalc import diffraction

logger = logging.getLogger(__name__)


@pytest.mark.parametrize('energy, material, reflection, expected', [
                         pytest.param(10e3, 'Si', (1, 1, 1),
                                      (11.402710639982848, 713.4828146545175)),
                         pytest.param(10e3, 'C', (1, 1, 1),
                                      (17.51878596767417, 427.8469911590626)),
                         pytest.param(9e3, 'Si', (2, 2, 2),
                                      (26.061879662584946, 233.3438968335023)),
                         pytest.param(9e3, 'C', (2, 2, 2),
                                      (41.98453277509927, 31.69504158241886))
                         ])
def test_get_geometry(energy, material, reflection, expected):
    # tested agains the old code
    res = diffraction.get_lom_geometry(energy, material, reflection)
    logger.debug('Expected: %s Received: %s', expected, res)
    assert np.isclose(expected, res).all()


@pytest.mark.parametrize('material_id, hkl, energy, expected', [
                         pytest.param('Si', (1, 1, 1), 10e3, 11.4027106399828),
                         pytest.param('Si', (1, 1, 1), 7e3, 16.40552015690347),
                         pytest.param('C', (1, 1, 1), 10e3, 17.51878596767417),
                         pytest.param('C', (2, 2, 2), 10e3, 37.01592462373143),
                         pytest.param('Si', (2, 2, 2), 7e3, 34.39310890947986),
                         pytest.param('C', (2, 2, 2), 9.8e3, 37.90277660759642)
                         ])
def test_bragg_angle(material_id, hkl, energy, expected):
    # tested agains the old code
    res = diffraction.bragg_angle(material_id, hkl, energy)
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
    res = diffraction.d_space(material_id, hkl)
    logger.debug('Expected: %s Received: %s', expected, res)
    assert np.isclose(expected, res)
