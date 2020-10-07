"""
Module for Beryllium Lens Calculations
"""
from periodictable import xsf
from periodictable import formula as ptable_formula
import numpy as np
import datetime
import os
import shutil
import pprint
from itertools import product


"""
If you are using the IMS IOC, each lens motor should have a
base record associated with it  i.e MFX:DIA:XFLS.
There should also be a binary record for each state,
that is just this base plus an arbitrary string i.e MFX:DIA:XFLS:OUT
"""

# We have sets of Be lenses with thicknesses:
LENS_RADII = [50e-6, 100e-6, 200e-6, 300e-6, 500e-6, 1000e-6, 1500e-6]


def getAttLen(E, material="Be", density=None):
    """ get the attenuation length (in meter) of material (default Si), if no
      parameter is given for the predefined energy;
      then T=exp(-thickness/att_len); E in keV"""
    att_len = float(xsf.attenuation_length(material, density=density,
                    energy=E))
    return att_len


def get_delta(E, material="Be", density=None):
    """
    Calculates delta for a given material at a given energy

    Parameters
    ----------
    E: number
        Beam Energy
    material : `str`
        Beryllium. The use of beryllium extends the range of operation
        of compound refractive lenses, improving transmission,
        aperture size, and gain
    density : TODO: find out what density is
    Returns
    --------
    delta: `int` TODO: not sure if int here or float?
    """
    delta = 1 - np.real(xsf.index_of_refraction(material, density=density,
                        energy=E))
    return delta


def calc_focal_length_for_single_lens(E, radius, material="Be", density=None):
    """
    Calculates the Focal Length for a single lens.

    Parameters
    ----------
    E: number
        Beam Energy
    radius : float TODO: is this float or no?
    material : `str`
        Beryllium. The use of beryllium extends the range of operation
        of compound refractive lenses, improving transmission,
        aperture size, and gain
    density : TODO: find out what density is

    Returns
    --------
    focal_length : `float`
        The focal length for a single lens
    """
    delta = get_delta(E, material, density)
    focal_length = radius / 2.0 / delta
    return focal_length


def calc_focal_length(E, lens_set, material="Be", density=None):
    """
    Calculates the Focal Length for certain lenses configuration and energy.

    Parameters
    -----------
    E: number
        Beam Energy
    lens_set : `list`
        [(numer1, lensthick1), (number2, lensthick2)...]
    material : `str`
        Beryllium. The use of beryllium extends the range of operation
        of compound refractive lenses, improving transmission,
        aperture size, and gain
    density : TODO: find out what density is

    Returns
    -------
    focal_length : `float`
    """
    num = []
    rad = []
    ftot_inverse = 0
    # if type(lens_set) is int:
    #     lens_set = getLensSet(lens_set)
    for i in range(len(lens_set) / 2):
        # TODO: why new code is (double //) range(len(lens_set)//2):
        num = lens_set[2 * i]
        rad = lens_set[2 * i + 1]
        if rad is not None:
            rad = float(rad)
            num = float(num)
            ftot_inverse += num / calcFocalLengthForSingleLens(
                            E, rad, material, density)
    return 1.0 / ftot_inverse


 def calc_beam_fwhm(self, E, lens_set, distance=None, source_distance=None, material="Be",
                       density=None, fwhm_unfocused=None, printsummary=True):
    """
    Calculates beam parameters for certain lenses configuration
    and energy at a given distance.
    Optionally some other parameters can be set
    Beamsize for certain lenses configuration and energy at a given distance

    Parameters
    -----------
    E: number
        Beam Energy
    lens_set : `list`
        [(numer1, lensthick1), (number2, lensthick2)...]
    distance: float
        Distance from the lenses to the sample is 3.852 m at XPP.
    source_distance: float
        Distance from source to lenses. This is about 160 m at XPP.
    material: str
        Beryllium. The use of beryllium extends the range of operation
        of compound refractive lenses, improving transmission,
        aperture size, and gain
    density: TODO: find out what density is
    fwhm_unfocused: float
        This is about 400 microns at XPP.
    printsummary: boolean
        Prints summary of parameters/calculations if True

    Returns
    -------
    float: Size FWHM
    """
    # Focal length for certain lenses configuration and energy
    focal_length = calc_focal_length(E, lens_set, material, density)
    
    # use lens makers equation to find distance to image of source
    if source_distance is not None:
        focal_length = 1 / (1 / focal_length - 1 / source_distance)
    
    lam = 1.2398 / E * 1e-9
    # the w parameter used in the usual formula is 2*sigma
    w_unfocused = fwhm_unfocused * 2 / 2.35
    # assuming gaussian beam divergence = w_unfocused/f we can obtain
    waist = lam / np.pi * focal_length / w_unfocused
    rayleigh_range = np.pi * waist ** 2 / lam
    size = waist * np.sqrt(1.0 + (distance - focal_length) ** 2.0 / rayleigh_range ** 2)
    size_fwhm = size * 2.35 / 2.0

    if printsummary:
        print("FWHM at lens   : %.3e" % (fwhm_unfocused))
        print("waist          : %.3e" % (waist))
        print("waist FWHM     : %.3e" % (waist * 2.35 / 2.0))
        print("rayleigh_range : %.3e" % (rayleigh_range))
        print("focal length   : %.3e" % (focal_length))
        print("size           : %.3e" % (size))
        print("size FWHM      : %.3e" % (size_fwhm))

    return size_fwhm


def find_energy(lens_set, distance=3.952, material="Be", density=None):
    """
    Finds the energy that would focus at a given distance (default = 4m)

    Parameters
    -----------
    lens_set : `list`
        [(numer1, lensthick1), (number2, lensthick2)...]
    distance : `float`
    material: str
        Beryllium. The use of beryllium extends the range of operation
        of compound refractive lenses, improving transmission,
        aperture size, and gain
    density: TODO: find out what density is
    
    usage findEnergy( (2,200e-6,4,500e-6) ,distance =4 )
  """
    Emin = 1.0
    Emax = 24.0
    E = (Emax + Emin) / 2.0
    absdiff = 100
    while absdiff > 0.0001:
        fmin = calc_focal_length(Emin, lens_set, material, density)
        fmax = calc_focal_length(Emax, lens_set, material, density)
        E = (Emax + Emin) / 2.0
        f = calc_focal_length(E, lens_set, material, density)
        if (distance < fmax) and (distance > f):
            Emin = E
        elif (distance > fmin) and (distance < f):
            Emax = E
        else:
            print("somehow failed ...")
            break
        absdiff = abs(distance - f)
    print("Energy that would focus at a distance of %.3f is %.3f" % (distance, E))
    s = calc_beam_fwhm(E, lens_set, distance, material, density)
    return E


def calc_distance_for_size(sizeFWHM, lens_set=None, E=None, fwhm_unfocused=None):
    size = sizeFWHM * 2.0 / 2.35
    f = calc_focal_length(E, lens_set, "Be", None)
    lam = 12.398 / E * 1e-10
    # the w parameter used in the usual formula is 2*sigma
    w_unfocused = fwhm_unfocused * 2 / 2.35
    # assuming gaussian beam divergence = w_unfocused/f we can obtain
    waist = lam / np.pi * f / w_unfocused
    rayleigh_range = np.pi * waist ** 2 / lam
    # bs = (size/waist)**2-1
    # if bs >= 0:
    distance = (
        np.sqrt((size / waist) ** 2 - 1) * np.asarray([-1.0, 1.0]) * rayleigh_range
    ) + f
    # else:
    # distance = nan
    #
    return distance


def calc_lens_set(
    Energy,
    sizeFWHM,
    distance,
    Nmax=12,
    maxeach=5,
    lensRadii=[100e-6, 200e-6, 300e-6, 500e-6, 1000e-6],
    fwhm_unfocused=0.0005,
    effrad0=None,
):
    Nums = product(*([list(range(maxeach + 1))] * len(lensRadii)))
    sets = []
    sizes = []
    effrads = []
    foclens = []
    for nums in Nums:
        lensset = []
        if sum(nums) <= Nmax and sum(nums) > 0:
            if effrad0 is None:
                teffradinv = 0
            else:
                teffradinv = 1 / effrad0
            for tn, tl in zip(nums, lensRadii):
                lensset += [tn, tl]
                teffradinv += tn / tl
            teffrad = np.round(1 / teffradinv, 6)
            # print teffrad
            if teffrad in effrads:
                ind = effrads.index(teffrad)
                # print nums
                # print sets[ind]
                # raw_input()

                if sum(sets[ind]) > sum(nums):
                    sets[ind] = nums
                else:
                    continue
            else:
                effrads.append(teffrad)
                sets.append(nums)
                sizes.append(
                    calcBeamFWHM(
                        Energy,
                        lensset + [1, effrad0],
                        distance=distance,
                        printit=False,
                        fwhm_unfocused=fwhm_unfocused,
                    )
                )
                foclens.append(calc_focal_length(Energy, lensset + [1, effrad0]))

    sizes = np.asarray(sizes)
    sets = np.asarray(sets)
    foclens = np.asarray(foclens)
    indsort = (np.abs(sizes - sizeFWHM)).argsort()

    return (
        sets[indsort, :],
        np.asarray(effrads)[indsort],
        sizes[indsort],
        foclens[indsort],
    )

def calc_lens_aperture_radius(radius, diskthickness=1e-3, apexdistance=30e-6):
    R0 = sqrt(radius * (diskthickness - apexdistance))
    return R0

def calc_trans_for_single_lens(
    E,
    radius,
    material="Be",
    density=None,
    fwhm_unfocused=900e-6,
    diskthickness=1.0e-3,
    apexdistance=30e-6,
):
    """ Calculates the transmission for a single lens.
      Usage : calcTransForSingleLens(E,radius,material="Be",density=None,fwhm_unfocused=800e-6,diskthickness=1.0e-3,apexdistance=30e-6):
  """
    delta = getDelta(E, material, density)
    mu = 1.0 / getAttLen(E, material="Be", density=None)
    s = fwhm_unfocused / 2.35482
    S = (s ** (-2.0) + 2.0 * mu / radius) ** (-0.5)
    R0 = sqrt(radius * (diskthickness - apexdistance))
    Trans = (
        (S ** 2 / s ** 2)
        * exp(-mu * apexdistance)
        * (1 - exp(-(R0 ** 2.0) / (2.0 * S ** 2)))
    )
    return Trans


def calc_trans_lens_set(
    E,
    lens_set,
    material="Be",
    density=None,
    fwhm_unfocused=900e-6,
    diskthickness=1.0e-3,
    apexdistance=30e-6,
):
    """ Calcultes the transmission of a lens set.
      usage : calcTrans(E,lens_set,material="Be",density=None,fwhm_unfocused=900e-6)
      There is latex document that explains the formula. Can be adapted to use different thicknesses for each lens,
      and different apex distances, but this would require changing the format of lens_set, which would mean changing
      a whole bunch of other programs too.
  """

    apexdistance_tot = 0
    radius_total_inv = 0
    radius_aperture = 1.0  # this is an ugly hack: the radius will never be bigger than 1m, so will always be overwritten
    if type(lens_set) is int:
        lens_set = getLensSet(lens_set)
    for i in range(len(lens_set) / 2):
        num = lens_set[2 * i]
        rad = lens_set[2 * i + 1]
        new_rad_ap = sqrt(rad * (diskthickness - apexdistance))
        radius_aperture = min(radius_aperture, new_rad_ap)
        radius_total_inv += num / rad
        apexdistance_tot += num * apexdistance
    radius_total = 1.0 / radius_total_inv
    equivalent_disk_thickness = radius_aperture ** 2 / radius_total + apexdistance_tot
    transtot = calc_trans_for_single_lens(
        E,
        radius_total,
        material,
        density,
        fwhm_unfocused,
        equivalent_disk_thickness,
        apexdistance_tot,
    )
    return transtot
