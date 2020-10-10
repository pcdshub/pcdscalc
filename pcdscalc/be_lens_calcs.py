'''
Module for Beryllium Lens Calculations
'''
import logging
import os
import shutil
from datetime import date
from itertools import product

import numpy as np
import xraydb as xdb


logger = logging.getLogger(__name__)

# We have sets of Be lenses with thicknesses:
LENS_RADII = [50e-6, 100e-6, 200e-6, 300e-6, 500e-6, 1000e-6, 1500e-6]

FWHM_SIGMA_CONVERSION = 2.35482004503
WAVELENGTH_PHOTON = 1.2398


def photon_to_wavelength(energy):
    '''
    Find the wavelength in micrometers, using the
    photon energy in electronvolts.
    The equation is approximately:
    λ[µm] = 1.2398 / E[eV]
    The photon energy at 1 μm wavelength,
    the wavelength of near infrared radiation, is approximately 1.2398 eV.

    Parameters
    ----------
    photon : `float`
        Photon energy in electronvolts
    Returns
    -------
    Wavelength in micrometers
    '''
    return WAVELENGTH_PHOTON / energy


def gaussian_sigma_to_fwhm(sigma):
    '''
    Converting between FWHM and sigma of a Gaussian function
    FWHM = 2.35482004503 * sigma

    https://brainder.org/2011/08/20/gaussian-kernels-convert-fwhm-to-sigma/

    Parameters
    ----------
    sigma : `float`

    Returns
    -------
    FWHM -  Full Width at the Half Maximum
    '''
    return FWHM_SIGMA_CONVERSION * sigma


def gaussian_fwhm_to_sigma(fwhm):
    '''
    Converting between FWHM and sigma of a Gaussian function
    sigma = FWHM / 2.35482004503

    https://brainder.org/2011/08/20/gaussian-kernels-convert-fwhm-to-sigma/

    Parameters
    ----------
    fwhm : `float`
        Full Witdh at the Half Maximum

    Returns
    -------
    sigma
    '''
    return fwhm / FWHM_SIGMA_CONVERSION


# TODO: this is a test file... take care of it
LENS_SET_FILE = os.path.dirname(__file__) + '/tests/test_lens_sets/lens_set'


def get_lens_set(set_number_top_to_bot, filename=LENS_SET_FILE):
    '''
    Get the lens set from the file provided

    Parameters
    ----------
    set_number_top_to_bot : `int`
        The Be lens holders can take 3 different sets that we usually set
        before experiments, this is to specify what number set
    filename : `str`
        File path of the lens_set file

    Returns
    -------
    lens_set : `list`
        [numer1, lensthick1, number2, lensthick2 ...]
    '''
    if not os.path.exists(filename):
        logger.error('Provided invalid path for lens set file: %s', filename)
        return
    if os.stat(filename).st_size == 0:
        logger.error('The file is empyt: %s', filename)
        return
    with open(filename) as f:
        sets = eval(f.read())
        if set_number_top_to_bot not in range(1, len(sets)):
            logger.error('Provided and invalid set_number_top_to_bottom %s, '
                         'please provide a number from 1 to %s ',
                         set_number_top_to_bot, len(sets))
            return
        print(sets[set_number_top_to_bot - 1])
        return sets[set_number_top_to_bot - 1]


def set_lens_set_to_file(sets_list_of_tuples, filename, make_backup=True):
    '''
    # The Be lens holders can take 3 different sets that we usually set before
    # few experiments so we only vent the relevant beamline section once.
    # We then store these sets into a config file that we can call from the
    #  methods you redefined. Later, we save this config file to the specific
    # experiment so users can make sure that they know which stack was used
    #  for there beamtime. Is that something we can get as well?

    usage :
    .. code-block:: python
        sets_list_of_tuples = [(3, 0.0001, 1, 0.0002),
                               (1, 0.0001, 1, 0.0003, 1, 0.0005),
                               (2, 0.0001, 1, 0.0005)]
        set_lens_set_to_file(sets_list_of_tuples, ../path/to/lens_set)
    '''
    # Make a backup with today's date
    if make_backup:
        backup_path = filename + str(date.today()) + '.bak'
        try:
            shutil.copyfile(filename, backup_path)
        except Exception as ex:
            logger.error('Something went wrong with copying the file %s', ex)
            pass
    with open(filename, 'w') as lens_file:
        # TODO: do we want to pprint.pformat() like in the old code?
        # lens_file.write(pprint.pformat(sets_list_of_tuples))
        lens_file.write(sets_list_of_tuples)


def get_att_len(energy, material="Be", density=None):
    '''
    Get the attenuation length (in meter) of a material

    The X-ray beam intensity I(x) at depth x in a material is a function
    of the attenuation coefficient mu, and can be calculated by the
    Beer-Lambert law.
    The Absorption Length (or Attenuation Length) is defined as the distance
    into a material where the x-ray beam intensity has decreased to a value
    of 1/e (~ 40%) of the incident beam intensity (Io)

    (1/e) = e^(-mu * x)
    ln(1/e) = ln(e^(-mu * x))
    1 = mu * x
    x = 1/mu

    usage :
    .. code-block:: python
        get_att_len(energy=8, material='Be')

    Parameters
    ----------
    energy : number
        Beam Energy given in KeV
    material : `str`
        Default - Beryllium.
        The use of beryllium extends the range of operation
        of compound refractive lenses, improving transmission,
        aperture size, and gain
    density : `float`
        Material density in g/cm^3

    Returns
    -------
    att_len : `numpy.float64`
        Attenuation length
    '''
    try:
        # xdb.material_my returns absorption length in 1/cm
        # and takes energy or array of energies in eV
        att_len = 1.0 / (xdb.material_mu(material, energy * 1.0e3,
                                         density=density)) * 1.0e-2

    except Exception as ex:
        logger.error('Get Attenuation Lenght error: %s', ex)
        return
    # TODO: if we return we'll get att_len == None, is this what we want?
    return att_len


def get_delta(energy, material="Be", density=None):
    '''
    Calculate delta for a given material at a given energy.
    Anomalous components of the index of refraction for a material,
    using the tabulated scattering components from Chantler

    Parameters
    ----------
    energy : number
        x-ray energy in KeV
    material : `str`
        Chemical formula (‘Fe2O3’, ‘CaMg(CO3)2’, ‘La1.9Sr0.1CuO4’)
        Default - Beryllium.
    density : `float`
        Material density in g/cm^3

    Returns
    -------
    delta : `numpy.float64`
        Real part of index of refraction
    '''
    # xray_delta_beta returns (delta, beta, atlen),
    # wehre delta : real part of index of refraction
    # takes x-ray energy in eV
    try:
        delta = xdb.xray_delta_beta(material,
                                    density=xdb.atomic_density(material),
                                    energy=energy * 1.0e3)[0]
    except Exception as ex:
        logger.error('Get Delta error: %s', ex)
        return
    # TODO: if we return we'll get delta == None, is this what we want?
    return delta


def calc_focal_length_for_single_lens(energy, radius, material="Be",
                                      density=None):
    '''
    Calculate the Focal Length for a single lens.

    Parameters
    ----------
    energy : number
        Beam Energy
    radius : `float`
    material : `str`
        Default - Beryllium.
        The use of beryllium extends the range of operation
        of compound refractive lenses, improving transmission,
        aperture size, and gain
    density : `float`
        Material density in g/cm^3

    Returns
    -------
    focal_length : `float`
        The focal length for a single lens
    '''
    delta = get_delta(energy, material, density)
    focal_length = radius / 2.0 / delta
    return focal_length


def calc_focal_length(energy, lens_set, material="Be", density=None):
    '''
    Calculate the Focal Length for certain lenses configuration and energy.

    Parameters
    ----------
    energy : number
        Beam Energy
    lens_set : `list`
        [numer1, lensthick1, number2, lensthick2 ...]
    material : `str`
        Default - Beryllium.
        The use of beryllium extends the range of operation
        of compound refractive lenses, improving transmission,
        aperture size, and gain
    density : `float`
        Material density in g/cm^3

    Returns
    -------
    focal_length : `float`
    '''
    f_tot_inverse = 0
    if isinstance(lens_set, int):
        lens_set = get_lens_set(lens_set)
    lens_set = (list(zip(lens_set[::2], lens_set[1::2])))
    for num, radius in lens_set:
        if radius is not None:
            fln = calc_focal_length_for_single_lens(energy, radius,
                                                    material, density)
            f_tot_inverse += num/fln

    return 1.0 / f_tot_inverse


def calc_beam_fwhm(energy, lens_set, distance=None, source_distance=None,
                   material="Be", density=None, fwhm_unfocused=500e-6,
                   printsummary=True):
    '''
    FWHM -  Full Width at the Half Maximum
    Calculates beam parameters for certain lenses configuration
    and energy at a given distance.
    Optionally some other parameters can be set

    Parameters
    ----------
    energy : number
        Beam Energy
    lens_set : `list`
        [numer1, lensthick1, number2, lensthick2...]
    distance : `float`
        Distance from the lenses to the sample is 3.852 m at XPP.
    source_distance : `float`
        Distance from source to lenses. This is about 160 m at XPP.
    material : `str`
        Beryllium. The use of beryllium extends the range of operation
        of compound refractive lenses, improving transmission,
        aperture size, and gain
    density : `float`
        Material density in g/cm^3
    fwhm_unfocused : `float`
        This is about 400 microns at XPP.
    printsummary : `bool`
        Prints summary of parameters/calculations if True

    Returns
    -------
    size_fwhm : `float`
    '''
    # Focal length for certain lenses configuration and energy
    focal_length = calc_focal_length(energy, lens_set, material, density)

    # use lens makers equation to find distance to image of source
    if source_distance is not None:
        focal_length = 1 / (1 / focal_length - 1 / source_distance)

    lam = photon_to_wavelength(energy) * 1e-9

    # the w parameter used in the usual formula is 2 * sigma
    w_unfocused = gaussian_fwhm_to_sigma(fwhm_unfocused) * 2

    # assuming gaussian beam divergence = w_unfocused/f we can obtain
    waist = lam / np.pi * focal_length / w_unfocused
    rayleigh_range = np.pi * waist ** 2 / lam
    size = waist * np.sqrt(1.0 + (distance - focal_length) ** 2.0 /
                           rayleigh_range ** 2)

    size_fwhm = gaussian_sigma_to_fwhm(size) / 2.0

    if printsummary:
        logger.info("FWHM at lens   : %.3e" % (fwhm_unfocused))
        logger.info("waist          : %.3e" % (waist))
        logger.info("waist FWHM     : %.3e" % (waist *
                                               FWHM_SIGMA_CONVERSION / 2.0))
        logger.info("rayleigh_range : %.3e" % (rayleigh_range))
        logger.info("focal length   : %.3e" % (focal_length))
        logger.info("size           : %.3e" % (size))
        logger.info("size FWHM      : %.3e" % (size_fwhm))

    return size_fwhm


def calc_distance_for_size(size_fwhm, lens_set=None, energy=None,
                           fwhm_unfocused=None):
    '''
    Calculate the distance for size

    Parameters
    ----------
    size_fwhm : `float`
    lens_set : `list`
        [numer1, lensthick1, number2, lensthick2...]
    energy : number
        Beam Energy
    fwhm_unfocused : `float`
        This is about 400 microns at XPP

    Returns
    -------
    distance : `float`
    '''
    size = gaussian_fwhm_to_sigma(size_fwhm) * 2.0
    focal_length = calc_focal_length(energy, lens_set, "Be", density=None)

    lam = photon_to_wavelength(energy) * 1e-9

    # the w parameter used in the usual formula is 2 * sigma
    w_unfocused = gaussian_fwhm_to_sigma(fwhm_unfocused) * 2.0

    # assuming gaussian beam divergence = w_unfocused/f we can obtain
    waist = lam / np.pi * focal_length / w_unfocused
    rayleigh_range = np.pi * waist ** 2 / lam

    distance = (np.sqrt((size / waist) ** 2 - 1) * np.asarray([-1.0, 1.0]) *
                rayleigh_range) + focal_length

    return distance


def calc_lens_aperture_radius(radius, disk_thickness=1.0e-3,
                              apex_distance=30e-6):
    '''
    Calculate the lens aperture raidus
    It is of importance to optimize which lens radius
    to use at a specific photon energy

    Usage
    .. code-block:: python
        calc_lens_aperture_radius(radius=4.0, disk_thickness=1e-3,
                                  apex_distance=30e-6)

    Parameters
    ----------
    radius : 'float`
    disk_thickness : `float`
        Default = 1.0e-3
    apex_distance : `float`
        Default = 30e-6

    Returns
    -------
    aperture_radius : `numpy.float64`
    '''
    aperture_radius = np.sqrt(radius * (disk_thickness - apex_distance))
    return aperture_radius


def calc_trans_for_single_lens(energy, radius, material="Be", density=None,
                               fwhm_unfocused=900e-6, disk_thickness=1.0e-3,
                               apex_distance=30e-6):
    '''
    Calculate the transmission for a single lens.

    usage :
    .. code-block:: python
        calc_trans_for_single_lens(energy, radius, material="Be",
                                   density=None, fwhm_unfocused=800e-6,
                                   disk_thickness=1.0e-3, apex_distance=30e-6)

    Parameters
    ----------
    energy : number
        Beam Energy
    radius : 'float`
    material : `str`
        Chemical formula. Default = 'Be'
    density : `float`
        Material density in g/cm^3
    fwhm_unfocused : `float`
        This is about 400 microns at XPP.
    disk_thickness : `float`
        Default = 1.0e-3
    apex_distance : `float`
        Default = 30e-6

    Returns
    -------
    transmission : `float`
        Transmission for a single lens
    '''
    # mu = mass attenuation coefficient?
    mu = 1.0 / get_att_len(energy, material=material, density=None)

    sigma = gaussian_fwhm_to_sigma(fwhm_unfocused)
    # TODO: what is S - Responsivity of a lens?
    S = (sigma ** (-2.0) + 2.0 * mu / radius) ** (-0.5)
    aperture_radius = calc_lens_aperture_radius(radius=radius,
                                                disk_thickness=disk_thickness,
                                                apex_distance=apex_distance)

    transmission = ((S ** 2 / sigma ** 2) * np.exp(-mu * apex_distance) *
                    (1 - np.exp(-(aperture_radius ** 2.0) / (2.0 * S ** 2))))
    return transmission


def calc_trans_lens_set(energy, lens_set, material="Be", density=None,
                        fwhm_unfocused=900e-6, disk_thickness=1.0e-3,
                        apex_distance=30e-6):
    '''
    Calculte the transmission of a lens set.
    These would allow us to estimate the total transmission of the lenses
    usage : calc_trans_lens_set(energy,lens_set,material="Be",density=None,
            fwhm_unfocused=900e-6)

    TODO: where is this document?
    There is latex document that explains the formula.
    Can be adapted to use different thicknesses for each lens,
    and different apex distances, but this would require changing
    the format of lens_set, which would mean changing
    a whole bunch of other programs too.

    usage :
    .. code-block:: python
        calc_trans_lens_set(energy, lens_set, material="Be", density=None,
                            fwhm_unfocused=900e-6)

    Parameters
    ----------
    energy : number
        Beam Energy
    lens_set : `list`
        [numer1, lensthick1, number2, lensthick2...]
    material : `str`
        Chemical formula. Default = 'Be'
    density : `float`
        Material density in g/cm^3
    fwhm_unfocused : `float`
        This is about 400 microns at XPP. Default = 900e-6
    disk_thickness : `float`
        Default = 1.0e-3
    apex_distance : `float`
        Default = 30e-6

    Returns
    -------
    transmission : `float`
        Transmission for a set of lens
    '''

    apex_distance_tot = 0
    radius_total_inv = 0
    # TODO: how can i fix this hack here:
    # this is an ugly hack: the radius will never be bigger than 1m,
    # so will always be overwritten
    radius_aperture = 1.0
    if type(lens_set) is int:
        lens_set = get_lens_set(lens_set)
    lens_set = (list(zip(lens_set[::2], lens_set[1::2])))
    for num, radius in lens_set:
        new_rad_ap = np.sqrt(radius * (disk_thickness - apex_distance))
        radius_aperture = min(radius_aperture, new_rad_ap)
        radius_total_inv += num / radius
        apex_distance_tot += num * apex_distance

    radius_total = 1.0 / radius_total_inv
    equivalent_disk_thickness = (radius_aperture ** 2 / radius_total +
                                 apex_distance_tot)

    transmission_total = calc_trans_for_single_lens(energy, radius_total,
                                                    material, density,
                                                    fwhm_unfocused,
                                                    equivalent_disk_thickness,
                                                    apex_distance_tot)
    return transmission_total


def calc_lens_set(energy, size_fwhm, distance, n_max=12, max_each=5,
                  lens_radii=[100e-6, 200e-6, 300e-6, 500e-6, 1000e-6],
                  fwhm_unfocused=0.0005, eff_rad0=None):
    '''
    Calculate lens set

    Parameters
    ----------
    energy : number
        Beam Energy
    size_fwhm : `float`
    distance : `float`
    n_max : `int`
    max_each : `int`
    lens_radii : `list`
    fwhm_unfocused : `float`
        This is about 400 microns at XPP. Default = 0.0005
    eff_rad0 : TODO: what is it?

    Returns
    -------
    TODO: what returns

    '''
    nums = product(*([list(range(max_each + 1))] * len(lens_radii)))
    sets = []
    sizes = []
    effrads = []
    foclens = []
    for num in nums:
        lens_set = []
        if sum(num) <= n_max and sum(num) > 0:
            if eff_rad0 is None:
                teffradinv = 0
            else:
                teffradinv = 1 / eff_rad0
            for tn, tl in zip(num, lens_radii):
                lens_set += [tn, tl]
                teffradinv += tn / tl
            teffrad = np.round(1 / teffradinv, 6)
            if teffrad in effrads:
                ind = effrads.index(teffrad)
                if sum(sets[ind]) > sum(num):
                    sets[ind] = num
                else:
                    continue
            elif teffrad is not None:
                effrads.append(teffrad)
                sets.append(num)
                size_fwhm = calc_beam_fwhm(energy, lens_set + [1, eff_rad0],
                                           distance=distance,
                                           source_distance=None,
                                           fwhm_unfocused=fwhm_unfocused,
                                           printsummary=False)
                sizes.append(size_fwhm)
                focal_length = calc_focal_length(energy,
                                                 lens_set + [1, eff_rad0])
                foclens.append(focal_length)

    sizes = np.asarray(sizes)
    sets = np.asarray(sets)
    foclens = np.asarray(foclens)
    indsort = (np.abs(sizes - size_fwhm)).argsort()

    lens_sets = (sets[indsort, :],
                 np.asarray(effrads)[indsort],
                 sizes[indsort],
                 foclens[indsort])

    return lens_sets


# TODO: ========== WE MIGHT NOT NEED THESE FUNCTIONS BELOW =================
# TODO: distance default might need to be changed
def find_radius(energy, distance=4.0, material="Be", density=None):
    '''
    Find the radius of curvature of the lens that would
    focus the energy at the distance
    usage:
    .. code-block:: python
        find_radius(energy, distance=4.0, material="Be", density=None)

    Parameters
    ----------
    energy : number
        Beam Energy
    distance : `float`
    material : `str`
        Chemical formula. Default = 'Be'
    density : `float`
        Material density in g/cm^3
    '''
    delta = get_delta(energy, material, density)
    radius = distance * 2 * delta
    return radius


def find_energy(lens_set, distance=3.952, material="Be", density=None):
    '''
    Find the energy that would focus at a given distance.

    usage :
    .. code-block:: python
        find_energy([2, 200e-6, 4, 500e-6], distance=4 )

    Parameters
    ----------
    lens_set : `list`
        [numer1, lensthick1, number2, lensthick2...]
    distance : `float`
    material : `str`
        Chemical formula. Default = 'Be'
    density : `float`
        Material density in g/cm^3

    Returns
    -------
    energy : float
        Energy
    '''
    energy_min = 1.0
    energy_max = 24.0
    energy = (energy_max + energy_min) / 2.0
    abs_diff = 100
    while abs_diff > 0.0001:
        focal_length_min = calc_focal_length(energy_min, lens_set,
                                             material, density)
        focal_length_max = calc_focal_length(energy_max, lens_set,
                                             material, density)
        energy = (energy_max + energy_min) / 2.0
        focal_length = calc_focal_length(energy, lens_set, material, density)
        if (distance < focal_length_max) and (distance > focal_length):
            energy_min = energy
        elif (distance > focal_length_min) and (distance < focal_length):
            energy_max = energy
        else:
            logger.error("somehow failed ...")
            break
        abs_diff = abs(distance - focal_length)
    logger.info("Energy that would focus at a distance of %.3f is %.3f"
                % (distance, energy))

    s = calc_beam_fwhm(energy, lens_set, distance=distance,
                       source_distance=None, material=material,
                       density=density)
    # TODO: s is not used, might have to remove it
    # but for now printing it out here
    logger.debug('s: %s', s)
    return energy


def find_z_pos(energy, lens_set, spot_size_fwhm, material="Be",
               density=None, fwhm_unfocused=800e-6):
    '''
    Find the two distances the Be lens needs to be at
    to get the spotsize in the chamber center

    usage :
    .. code-block:: python
        find_z_pos(energy, lens_set, spotsizefwhm, material="Be",
                   density=None, fwhm_unfocussed=200e-6)

    Parameters
    ----------
    energy : number
        Beam Energy
    lens_set : `list`
        [numer1, lensthick1, number2, lensthick2...]
    spot_size_fwhm :
    material : `str`
        Chemical formula. Default = 'Be'
    density : `float`
        Material density in g/cm^3
    fwhm_unfocused : `float`
        This is about 400 microns at XPP. Default = 800e-6
    Returns
    -------
    z_position : `tuple`
        (z1, z2)
    '''
    focal_length = calc_focal_length(energy, lens_set, material, density)

    lam = photon_to_wavelength(energy) * 1e-9
    # the w parameter used in the usual formula is 2 * sigma
    w_unfocused = gaussian_fwhm_to_sigma(fwhm_unfocused) * 2
    waist = lam / np.pi * focal_length / w_unfocused
    rayleigh_range = np.pi * waist ** 2 / lam

    logger.info("waist          : %.3e" % waist)
    logger.info("waist FWHM     : %.3e" % (waist *
                                           FWHM_SIGMA_CONVERSION / 2.0))
    logger.info("rayleigh_range : %.3e" % rayleigh_range)
    logger.info("focal length   : %.3e" % focal_length)

    w = gaussian_fwhm_to_sigma(spot_size_fwhm) * 2
    delta_z = rayleigh_range * np.sqrt((w / waist) ** 2 - 1)
    z1 = focal_length - delta_z
    z2 = focal_length + delta_z
    z_position = (z1, z2)
    return z_position
