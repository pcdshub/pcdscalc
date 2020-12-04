"""Module that holds common calculations."""
from .constants import alias

# TODO: find a better number, it should be None here...
_manual_energy = 9.8e3

# wavelength/energy = 12398.4 (A) / E (eV)
WAVELENGTH_TO_ENERGY_LAMBDA = 12398.4


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


def get_energy(energy=None):
    """
    Get working energy.

    If energy passed in, return it, otherwise return manual_energy if set, or
    machine energy otherwise.

    Parameters
    ----------
    energy : number

    Returns
    -------
    en : number
        Photon energy in eV.
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
    return en
