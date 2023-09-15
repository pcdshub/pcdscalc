"""
Module that handles calculations for diffraction/crystallography.

Assume all the calculations that expect energy as input will
require energy in eV.
"""
import numpy as np
import xraylib

import pcdscalc.constants as cst

from .common import asind, cosd, energy_to_wavelength, sind
from .constants import const, units


def get_lom_geometry(energy, material_id, reflection):
    """
    Calculate the Large Offset Monochromator's crystal geometry.

    Parameters
    ----------
    energy : number
        Photon energy in eV.
    material_id : str
        Chemical formula. E.g.: `Si`.
    reflection : tuple
        The reflection. E.g.: `(1,1,1)`.

    Returns
    -------
    thm, zm : tuple
    """
    th = np.radians(bragg_angle(material_id, reflection, energy))
    zm = 300 / np.tan(2 * th)
    thm = np.rad2deg(th)
    return thm, zm


def bragg_angle(material_id, hkl, energy):
    """
    Compute the Bragg angle in deg.

    Computes the bragg angle of the specified material, reflection and photon
    energy.

    Parameters
    ----------
    material_id : str
        Chemical formula. Defaults to `Si`.
    hkl : tuple
        The reflection indices. Defaults to `(1,1,1)`.
    energy : number
        The photon energy in eV.

    Returns
    -------
    theta : number
        Theta in degrees.
    """
    material_id = cst.chemical_name_to_formula.get(material_id, material_id)
    d = d_space(material_id, hkl)
    theta = asind(energy_to_wavelength(energy) / 2 / d)
    return theta


def d_space(material_id, hkl):
    """
    Compute the d spacing (m) of the specified material and reflection.

    The d-spacing can be described as the distance between planes of atoms
    that give rise to diffraction peaks. Each peak in a diffractogram results
    from a corresponding d-spacing. The planes of atoms can be referred to a
    3D coordinate system and so can be described as a direction within the
    crystal. So d-spacing as well as having a dimension, usually quoted in
    Ångstroms, can be labelled with a plane direction hkl.

    Parameters
    ----------
    material_id : str
        Chemical fomula. E.g.: `Si`
    hkl : tuple
        Miller Indices, the reflection. E.g.: `(1,1,1)`

    Returns
    -------
    d : number
        The lattice plane spacing d in Bragg's Law for the chosen plane
        indicated by the Miller indices.
    """
    material_id = cst.chemical_name_to_formula.get(material_id, material_id)
    h_index, k_index, l_index = hkl
    lp = cst.lattice_parameters[material_id]
    # a, b, c in angstroms
    # alpha, beta, gamma in degrees
    a, b, c, alpha, beta, gamma = lp
    a = a / units["ang"]
    b = b / units["ang"]
    c = c / units["ang"]

    cos_alpha = cosd(alpha)
    cos_beta = cosd(beta)
    cos_gamma = cosd(gamma)
    sin_alpha = sind(alpha)
    sin_beta = sind(beta)
    sin_gamma = sind(gamma)

    inv_d_sqr = (
        1 / (1 + 2 * cos_alpha * cos_beta * cos_gamma - cos_alpha ** 2
             - cos_beta ** 2 - cos_gamma ** 2)
        * (
            h_index ** 2 * sin_alpha ** 2 / a ** 2
            + k_index ** 2 * sin_beta ** 2 / b ** 2
            + l_index ** 2 * sin_gamma ** 2 / c ** 2
            + 2 * h_index * k_index *
            (cos_alpha * cos_beta - cos_gamma) / a / b
            + 2 * k_index * l_index *
            (cos_beta * cos_gamma - cos_alpha) / b / c
            + 2 * h_index * l_index *
            (cos_alpha * cos_gamma - cos_beta) / a / c
        )
    )
    d = inv_d_sqr ** -0.5
    return d


def form_factor(material_id, twotheta, energy):
    """
    Returns the atomic form factor for Rayleigh scattering

    Parameters
    ----------
    material_id: str
    twotheta: float
        scattering angle in degrees
    energy: float
    """
    material_id = cst.chemical_name_to_formula.get(material_id, material_id)
    z = cst.element_z[material_id]
    q = xraylib.MomentTransf(energy*1e-3, np.deg2rad(twotheta/2))
    f = xraylib.FF_Rayl(z, q)
    return f


def structure_factor(material_id, f, hkl, z=None):
    material_id = cst.chemical_name_to_formula.get(material_id, material_id)
    lattice = cst.lattice_type[material_id]
    h, k, l = hkl  # noqa: E741

    if lattice == 'fcc':
        x1 = np.exp(-1j*np.pi * (k+l))
        x2 = np.exp(-1j*np.pi * (h+l))
        x3 = np.exp(-1j*np.pi * (h+k))
        F = f * (1 + x1 + x2 + x3)
    elif lattice == 'bcc':
        F = f * (1 + np.exp(-1j*np.pi * (h+k+l)))
    elif lattice == 'cubic':
        F = f
    elif lattice == 'diamond':
        x1 = np.exp(-1j*np.pi * (k+l))
        x2 = np.exp(-1j*np.pi * (h+l))
        x3 = np.exp(-1j*np.pi * (h+k))
        x4 = np.exp(-1j*2*np.pi * (h/4.0 + k/4.0 + l/4.0))
        F = f * (1 + x1 + x2 + x3) * (1 + x4)
    elif lattice == 'rhomb':
        z = cst.latticeParamRhomb[material_id]
        F = f * (1 + np.exp(2*1j*np.pi * (h+k+l) * z))
    elif lattice == 'tetr':
        F = f
    elif lattice == 'hcp':
        F = f * (1 + np.exp(2*1j*np.pi * (h/3.0 + 2*k/3.0 + l/2.0)))
    return F


def unit_cell_volume(material_id):
    """
    Unit cell volume
    """
    material_id = cst.chemical_name_to_formula.get(material_id, material_id)
    lp = cst.lattice_parameters[material_id]
    # a, b, c in angstroms
    # alpha, beta, gamma in degrees
    a, b, c, alpha, beta, gamma = lp
    a = a / units["ang"]
    b = b / units["ang"]
    c = c / units["ang"]
    cos_alpha = cosd(alpha)
    cos_beta = cosd(beta)
    cos_gamma = cosd(gamma)
    r = 1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 \
        + 2 * cos_alpha * cos_beta * cos_gamma
    volume = a * b * c * np.sqrt(r)
    return volume


def darwin_width(material_id, hkl, energy, T=293):
    """
    Computes the Darwin width for a specified crystal reflection (degrees)

    Parameters
    ----------
    material_id : str
        Chemical fomula. E.g.: `Si`
    hkl : tuple
        Miller Indices, the reflection. E.g.: `(1,1,1)`
    E : float
        Photon energy in eV
    T : float
        Temperature in Kelvin.

    Returns
    -------
    darwin_width : float
        Darwin width in deg of the reflection
    """
    material_id = cst.chemical_name_to_formula.get(material_id, material_id)
    lam = energy_to_wavelength(energy)
    theta = bragg_angle(material_id, hkl, energy)

    f = form_factor(material_id, 2*theta, energy)
    F = structure_factor(material_id, f, hkl)
    vol = unit_cell_volume(material_id)
    dw = (2 * const['eRad'] * lam**2 * np.abs(F)) \
        / (np.pi * vol * sind(2*theta)) / units['rad']
    return dw
