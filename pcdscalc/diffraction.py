"""
Module that handles calculations for diffraction/crystallography.

Assume all the calculations that expect energy as input will
require energy in eV.
"""
import numpy as np

from .constants import lattice_parameters, units, alias

# wavelength/energy = 12398.4 (A) / E(eV)
WAVELENGTH_TO_ENERGY_LAMBDA = 12398.4


def energy_to_wavelength(energy):
    """
    Compute photon wavelength in m.

    Parameters
    ----------
    energy : number
        Photon energy in eV.

    Returns
    -------
    wavelength : float
        Wavelength [m].
    """
    return (WAVELENGTH_TO_ENERGY_LAMBDA / energy) / units["ang"]


def wavelength_to_energy(wavelength):
    """
    Compute photon energy in eV.

    Parameters
    ----------
    wavelength : number
        The photon wavelength in m.

    Returns
    -------
    energy : number
        Photon Energy in eV.
    """
    return WAVELENGTH_TO_ENERGY_LAMBDA / (wavelength * units["ang"])


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
    material_id = alias.get(material_id, material_id)
    d = d_space(material_id, hkl)
    theta = np.arcsin(energy_to_wavelength(energy) / 2 / d)
    return np.degrees(theta)


def d_space(material_id, hkl):
    """
    Compute the d spacing (m) of the specified material and reflection.

    Parameters
    ----------
    material_id : str
        Chemical fomula. E.g.: `Si`
    hkl : tuple
        Miller Indices, the reflection. E.g.: `(1,1,1)`

    Returns
    -------
    d : number
        Inverse d_space squared.
    """
    material_id = alias.get(material_id, material_id)
    h_index, k_index, l_index = hkl
    lp = lattice_parameters[material_id]
    # a, b, c in angstroms
    # alpha, beta, gamma in degrees
    a, b, c, alpha, beta, gamma = lp
    a = a / units["ang"]
    b = b / units["ang"]
    c = c / units["ang"]
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    cos_gamma = np.cos(gamma)
    sin_alpha = np.sin(alpha)
    sin_beta = np.sin(beta)
    sin_gamma = np.sin(gamma)

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
