Beryllium Lens Calculations
===========================
This module contains multiple calculations to assist adjusting the Be Lenses to focus the beam.

Here is a quick example of how you would get started with this module.

>>> from pcdscalc import be_lens_calcs as be

Configure the defaults used for calculations:

>>> be.configure_defaults(distance=3.852, fwhm_unfocused=800e-6)
>>> be.DISTANCE
3.852
>>> be.FWHM_UNFOCUSED
0.0008

.. note::

   The following defaults will be used if not configured with :meth:`pcdscalc.be_lens_calcs.configure_defaults` function:

   .. code::

      # full width at half maximum unfocused
      FWHM_UNFOCUSED = 500e-6

      # Disk Thickness
      DISK_THICKNESS = 1.0e-3

      # Apex of the lens
      APEX_DISTANCE = 30e-6

      # Distance from the lenses to the sample
      DISTANCE = 4.0

      # Atomic symbol for element
      MATERIAL = 'Be'

      # Set of Be lenses with thicknesses:
      LENS_RADII = [50e-6, 100e-6, 200e-6, 300e-6, 500e-6, 1000e-6, 1500e-6, 2000e-6, 3000e-6]

Configure the path to the lens_set file that will be used in multiple calculations:

>>> be.configure_lens_set_file('/path/for/current/be_lens_file')

Store sets in the `be_lens_file`:

>>> sets_list = [[3, 0.0001, 1, 0.0002],
                 [1, 0.0001, 1, 0.0003],
                 [2, 0.0001, 1, 0.0005]]
>>> set_lens_set_to_file(sets_list)

Get the first set that was stored in the `be_lens_file`:

>>> be.get_lens_set(1)
[3, 0.0001, 1, 0.0002]

Get the whole stack of lens sets from the `be_lens_file`:

>>> be.get_lens_set(None, get_all=True)
[[3, 0.0001, 1, 0.0002],
 [1, 0.0001, 1, 0.0003, 1, 0.0005],
 [2, 0.0001, 1, 0.0005]]

.. note::

   The Be lens holders can take 3 different sets that could be set before experiments so that only the relevant beamline section is vented once. These sets can be stored in a file with the :meth:`pcdscalc.be_lens_cals.set_lens_set_to_file` function, and they can be accessed in other calculations with :func:`pcsdcalc.be_lens_cals.get_lens_set`. The file containing the lens sets could be then saved to a specific experiment so users know which stack was used for the beamtime.


For more examples please look at each individual function's `Example` section.

.. currentmodule:: pcdscalc.be_lens_calcs

.. automodule:: pcdscalc.be_lens_calcs
   :members:
   :undoc-members:
   :show-inheritance:
