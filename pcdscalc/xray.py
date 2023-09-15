import numpy as np
import periodictable
import xraylib

import pcdscalc.constants as cst

from .common import get_energy
from .constants import const, units


def molecular_mass(material_id):
    """Returns the molecular mass of a chemical formula in g"""
    material_id = cst.chemical_name_to_formula.get(material_id, material_id)
    form = periodictable.formula(material_id)
    mass = 0
    for elem, atoms in form.atoms.items():
        element_id = elem.symbol
        mass += cst.atomic_mass[element_id]*atoms
    return mass


def cs_photo(material_id, energy=None):
    """
    Returns the total photoabsorption cross section in m^2
    NOTE: This is per molecule if chemical formula given

    Parameters
    ----------
    material_id: str
        Chemical fomula : 'Si'
    energy: float

    """
    material_id = cst.chemical_name_to_formula.get(material_id, material_id)
    energy = get_energy(energy=energy)
    energy_kev = energy/1000.  # for xraylib
    form = periodictable.formula(material_id)
    CS = 0
    for elem, atoms in form.atoms.items():
        element_id = elem.symbol
        cs_photo = xraylib.CS_Photo(cst.element_z[element_id], energy_kev)
        CS += cs_photo * atoms * cst.atomic_mass[element_id]/const['NA']/units['cm']**2
    return CS


def cs_compt(material_id, energy=None):
    """
    Returns the total Compton (inelastic) cross section in m^2

    Parameters
    ----------
    material_id: str
        Chemical fomula : 'Si'
    energy: float
    """
    material_id = cst.chemical_name_to_formula.get(material_id, material_id)
    energy = get_energy(energy=energy)
    energy_kev = energy/1000.  # for xraylib
    form = periodictable.formula(material_id)
    CS = 0
    for elem, atoms in form.atoms.items():
        element_id = elem.symbol
        cs_compt = xraylib.CS_Compt(cst.element_z[element_id], energy_kev)
        CS += cs_compt * atoms * cst.atomic_mass[element_id]/const['NA']/units['cm']**2
    return CS


def attenuation_length(material_id, energy=None, density=None, compton=True):
    """
    Computes the attenuation length of a solid in m (photoabsoption and Compton cross sections)

    Parameters
    ----------
    material_id: str
        Chemical fomula : 'Si'
    energy: float
        E is photon energy in eV
    density: float
        Material density in g/cm^3
        If no density is specified will use default value
    """
    material_id = cst.chemical_name_to_formula.get(material_id, material_id)
    energy = get_energy(energy=energy)
    if density is None:
        density = cst.density[material_id]
    mu_rho = cs_photo(material_id, energy)
    if compton:
        mu_rho += cs_compt(material_id, energy)
    mu_rho = const['NA']*mu_rho / molecular_mass(material_id)*density*units['cm']**3
    att_length = 1.0 / mu_rho
    return att_length


def transmission(material_id, thickness , energy=None, density=None, compton=True):
    """
    Computes the transmission of a solid (photoabsoption and Compton cross sections)

    Parameters
    ----------
    material_id: str
        Chemical fomula : 'Si'
    thickness: float
        Material thickness in m
    energy: float
        E is photon energy in eV or keV
    density: float
        Material density in g/cm^3
        If no density is specified will use default value
    """
    energy = get_energy(energy=energy)
    material_id = cst.chemical_name_to_formula.get(material_id, material_id)
    att_length = attenuation_length(material_id, energy, density, compton=compton)
    transmission = np.exp(-thickness/att_length)
    return transmission


def transmission_no_compton(material_id, thickness , energy=None, density=None):
    transmission(material_id, thickness , energy=None, density=None, compton=False)
