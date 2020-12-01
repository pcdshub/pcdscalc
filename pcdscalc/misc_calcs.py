import numpy as np
from .be_lens_calcs import alias

# TODO: find a better number
_manual_energy = 9.8


def set_energy(energy):
    """
    Set a global, manual, working energy for doing offline calcs.
    """
    global _manual_energy
    _manual_energy = energy


def check_id(material_id):
    """Check to see if you are using an alias. Return the chemical formula."""
    try:
        return alias[material_id]
    except Exception:
        return material_id


def cosd(angle):
    """Cos of an angle specified in degrees."""
    arad = np.deg2rad(angle)
    x = np.cos(arad)
    return x


def sind(angle):
    """Sin of an angle specified in degrees."""
    arad = np.deg2rad(angle)
    x = np.sin(arad)
    return x


def asind(angle):
    """Arcsin in degrees. Closed interval of [-pi/2, pi/2]."""
    a = np.arcsin(angle)
    a = np.rad2deg(a)
    return a


def get_energy(energy=None, correct_ev=True):
    """
    Get working energy.

    Convert to eV if correct_ev == True
    If energy passed in, return it, otherwise return manual_energy if set, or
    machine energy otherwise.

    Parameters
    ----------
    energy : number
        Photon energy in eV or keV.
    correct_ev : bool
        Indicates if energy should be converted to eV.

    Returns
    -------
    en : number
        Photon energy in eV or keV.
    """
    en = None
    if energy is not None:
        en = energy
    elif _manual_energy is not None:
        en = _manual_energy
    else:
        # en = _ePv.get()
        # TODO: we need a way of getting the actual energy here.
        pass
    if correct_ev:
        en = to_ev(en)
        pass
    return en


def to_ev(energy):
    """Return photon energy in eV if specified in eV or keV."""
    if energy < 100:
        energy = energy * 1000.0
    return energy * 1.0


def energy_to_wavelength(energy, sub_100_energy=False):
    """
    Compute photon wavelength in m.

    Parameters
    ----------
    energy : number
        Photon energy in eV or keV.
    sub_100_energy : bool
        Set `sub_100_energy` to False if working at sub-100 eV energies.
    """
    if sub_100_energy:
        energy = energy
    else:
        energy = to_ev(energy)
    lam = (12398.4 / energy) / units["ang"]
    return lam


def get_geometry(energy, material_id, reflection):
    """
    Calculate the geometry.

    Parameters
    ----------
    energy : number
        Photon energy in .
    material_id : str
        Chemical formula.
    reflection : tuple
        The reflection. E.g.: `(1,1,1)`.

    Returns
    -------
    thm, zm : tuple
    """
    th = bragg_angle(material_id, reflection, energy) * np.pi / 180
    zm = 300 / np.tan(2 * th)
    thm = np.rad2deg(th)
    return thm, zm


def bragg_angle(material_id, hkl, energy=None):
    """
    Compute the Bragg angle in deg.

    Computes the bragg angle of the specified material, reflection and photon
    energy.

    Parameters
    ----------
    material_id : str
        Chemical formula. Defaults to `Si`.
    hkl : tuple
        The reflection. Defaults to `(1,1,1)`.
    energy : number
        The photon energy in eV or keV.
    """
    material_id = check_id(material_id)
    energy = get_energy(energy=energy, correct_ev=True)
    d = d_space(material_id, hkl)
    theta = asind(energy_to_wavelength(energy) / 2 / d)
    return theta


def wavelength_to_energy(wavelength):
    """
    Compute photon energy in eV.

    Parameters
    ----------
    wavelength : number
        The photon wavelength in m.
    """
    energy = 12398.4 / (wavelength * units["ang"])
    return energy


def d_space(material_id, hkl):
    """
    Compute the d spacing (m) of the specified material and reflection.

    Parameters
    ----------
    material_id : str
        Chemical fomula. E.g.: `Si`
    hkl : tuple
        Miller Indices, the reflection. E.g.: `(1,1,1)`
    """
    material_id = check_id(material_id)
    h_index = hkl[0]
    k_index = hkl[1]
    l_index = hkl[2]

    lp = lattice_parameters[material_id]
    a = lp[0] / units["ang"]
    b = lp[1] / units["ang"]
    c = lp[2] / units["ang"]
    alpha = lp[3]
    beta = lp[4]
    gamma = lp[5]

    ca = cosd(alpha)
    cb = cosd(beta)
    cg = cosd(gamma)
    sa = sind(alpha)
    sb = sind(beta)
    sg = sind(gamma)

    inv_d_sqr = (
        1 / (1 + 2 * ca * cb * cg - ca ** 2 - cb ** 2 - cg ** 2)
        * (
            h_index ** 2 * sa ** 2 / a ** 2
            + k_index ** 2 * sb ** 2 / b ** 2
            + l_index ** 2 * sg ** 2 / c ** 2
            + 2 * h_index * k_index * (ca * cb - cg) / a / b
            + 2 * k_index * l_index * (cb * cg - ca) / b / c
            + 2 * h_index * l_index * (ca * cg - cb) / a / c
        )
    )
    d = inv_d_sqr ** -0.5
    return d


# Crystal Lattice parameters (a, b, c, alpha, beta, gamma)
# a,b,c in angstroms
# alpha, beta, gamma in degrees
lattice_parameters = {
    "H": (3.75, 3.75, 6.12, 90, 90, 120),
    "He": (3.57, 3.57, 5.83, 90, 90, 120),
    "Li": (3.491, 3.491, 3.491, 90, 90, 90),
    "Be": (2.2866, 2.2866, 3.5833, 90, 90, 120),
    "B": (5.06, 5.06, 5.06, 58.06, 58.06, 58.06),
    "C": (3.567, 3.567, 3.567, 90, 90, 90),
    "N": (5.66, 5.66, 5.66, 90, 90, 90),
    "Ne": (4.66, 4.66, 4.66, 90, 90, 90),
    "Na": (4.225, 4.225, 4.225, 90, 90, 90),
    "Mg": (3.21, 3.21, 5.21, 90, 90, 120),
    "Al": (4.05, 4.05, 4.05, 90, 90, 90),
    "Si": (5.4310205, 5.4310205, 5.4310205, 90, 90, 90),
    "Ar": (5.31, 5.31, 5.31, 90, 90, 90),
    "K": (5.225, 5.225, 5.225, 90, 90, 90),
    "Ca": (5.58, 5.58, 5.58, 90, 90, 90),
    "Sc": (3.31, 3.31, 5.27, 90, 90, 120),
    "Ti": (2.95, 2.95, 4.68, 90, 90, 120),
    "V": (3.03, 3.03, 3.03, 90, 90, 90),
    "Cr": (2.88, 2.88, 2.88, 90, 90, 90),
    "Fe": (2.87, 2.87, 2.87, 90, 90, 90),
    "Co": (2.51, 2.51, 4.07, 90, 90, 120),
    "Ni": (3.52, 3.52, 3.52, 90, 90, 90),
    "Cu": (3.61, 3.61, 3.61, 90, 90, 90),
    "Zn": (2.66, 2.66, 4.95, 90, 90, 120),
    "Ge": (5.658, 5.658, 5.658, 90, 90, 90),
    "As": (4.1018, 4.1018, 4.1018, 54.554, 54.554, 54.554),
    "Kr": (5.64, 5.64, 5.64, 90, 90, 90),
    "Rb": (5.585, 5.585, 5.585, 90, 90, 90),
    "Sr": (6.08, 6.08, 6.08, 90, 90, 90),
    "Y": (3.65, 3.65, 5.73, 90, 90, 120),
    "Zr": (3.23, 3.23, 5.15, 90, 90, 120),
    "Nb": (3.3, 3.3, 3.3, 90, 90, 90),
    "Mo": (3.15, 3.15, 3.15, 90, 90, 90),
    "Tc": (2.74, 2.74, 4.4, 90, 90, 120),
    "Ru": (2.71, 2.71, 4.28, 90, 90, 120),
    "Rh": (3.8, 3.8, 3.8, 90, 90, 90),
    "Pd": (3.89, 3.89, 3.89, 90, 90, 90),
    "Ag": (4.09, 4.09, 4.09, 90, 90, 90),
    "Cd": (2.98, 2.98, 5.62, 90, 90, 120),
    "In": (3.25, 3.25, 4.95, 90, 90, 90),
    "Sn": (6.49, 6.49, 6.49, 90, 90, 90),
    "Sb": (4.4898, 4.4898, 4.4898, 57.233, 57.233, 57.233),
    "Xe": (6.13, 6.13, 6.13, 90, 90, 90),
    "Cs": (6.045, 6.045, 6.045, 90, 90, 90),
    "Ba": (5.02, 5.02, 5.02, 90, 90, 90),
    "Ce": (5.16, 5.16, 5.16, 90, 90, 90),
    "Eu": (4.58, 4.58, 4.58, 90, 90, 90),
    "Gd": (3.63, 3.63, 5.78, 90, 90, 120),
    "Tb": (3.6, 3.6, 5.7, 90, 90, 120),
    "Dy": (3.59, 3.59, 5.65, 90, 90, 120),
    "Ho": (3.58, 3.58, 5.62, 90, 90, 120),
    "Er": (3.56, 3.56, 5.59, 90, 90, 120),
    "Tm": (3.54, 3.54, 5.56, 90, 90, 120),
    "Yb": (5.45, 5.45, 5.45, 90, 90, 90),
    "Lu": (3.5, 3.5, 5.55, 90, 90, 120),
    "Hf": (3.19, 3.19, 5.05, 90, 90, 120),
    "Ta": (3.3, 3.3, 3.3, 90, 90, 90),
    "W": (3.16, 3.16, 3.16, 90, 90, 90),
    "Re": (2.76, 2.76, 4.46, 90, 90, 120),
    "Os": (2.74, 2.74, 4.32, 90, 90, 120),
    "Ir": (3.84, 3.84, 3.84, 90, 90, 90),
    "Pt": (3.92, 3.92, 3.92, 90, 90, 90),
    "Au": (4.08, 4.08, 4.08, 90, 90, 90),
    "Tl": (3.46, 3.46, 5.52, 90, 90, 120),
    "Pb": (4.95, 4.95, 4.95, 90, 90, 90),
    "Bi": (4.7236, 4.7236, 4.7236, 57.35, 57.35, 57.35),
    "Po": (3.34, 3.34, 3.34, 90, 90, 90),
    "Ac": (5.31, 5.31, 5.31, 90, 90, 90),
    "Th": (5.08, 5.08, 5.08, 90, 90, 90),
    "Pa": (3.92, 3.92, 3.24, 90, 90, 90),
    "ZnSe": (5.6676, 5.6676, 5.6676, 90, 90, 90),
    "ZnTe": (6.101, 6.101, 6.101, 90, 90, 90),
    "CdS": (5.832, 5.832, 5.832, 90, 90, 90),
    "CdSe": (6.05, 6.05, 6.05, 90, 90, 90),
    "CdTe": (6.477, 6.477, 6.477, 90, 90, 90),
    "BN": (3.615, 3.615, 3.615, 90, 90, 90),
    "GaSb": (6.0954, 6.0954, 6.0954, 90, 90, 90),
    "GaAs": (5.65315, 5.65315, 5.65315, 90, 90, 90),
    "GaMnAs": (5.65, 5.65, 5.65, 90, 90, 90),
    "GaP": (5.4505, 5.4505, 5.4505, 90, 90, 90),
    "InP": (5.86875, 5.86875, 5.86875, 90, 90, 90),
    "InAs": (6.05838, 6.05838, 6.05838, 90, 90, 90),
    "InSb": (6.47877, 6.47877, 6.47877, 90, 90, 90),
    "LaMnO3": (5.531, 5.602, 7.742, 90, 90, 90),
    "LaAlO3": (5.377, 5.377, 5.377, 60.13, 60.13, 60.13),
    "La0.7Sr0.3MnO3": (5.4738, 5.4738, 5.4738, 60.45, 60.45, 60.45),
    "Gd3Ga5O12": (12.383, 12.383, 12.383, 90, 90, 90),
}

# define units and constants
units = {
    "fm": 1e15,
    "pm": 1e12,
    "ang": 1e10,
    "nm": 1e9,
    "um": 1e6,
    "mm": 1e3,
    "cm": 1e2,
    "km": 1e-3,
    "kHz": 1e-3,
    "MHz": 1e-6,
    "GHz": 1e-9,
    "THz": 1e-12,
    "PHz": 1e-15,
    "inch": 39.370079,
    "mile": 0.000621,
    "ft": 3.28084,
    "yard": 1.093613,
    "mil": 39.370079 * 1000,
    "barn": 1e28,
    "fs": 1e15,
    "ps": 1e12,
    "ns": 1e9,
    "us": 1e6,
    "ms": 1e3,
    "min": 1 / 60.0,
    "hour": 1 / 3600.0,
    "day": 1 / (3600 * 24.0),
    "mdeg": 1e3,
    "udeg": 1e6,
    "ndeg": 1e9,
    "rad": np.pi / 180,
    "mrad": np.pi / 180 * 1e3,
    "urad": np.pi / 180 * 1e6,
    "nrad": np.pi / 180 * 1e9,
    "asec": 3600,
    "amin": 60,
    "g": 1e3,
    "eV": 6.2415e18,
    "erg": 1e7,
    "cal": 0.239,
    "mJ": 1e3,
    "uJ": 1e6,
    "nJ": 1e9,
    "pJ": 1e9,
    "Torr": 7.5006e-3,
}
