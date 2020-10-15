Beryllium Lens Calculations
===========================
This module contains multiple calculations to assist adjusting the Be Lenses to focus the beam.

Here is a quick example of how you would get started with this module. 

>>> from pcdscalc import be_lens_calcs as be

Configure the path to the lens_set file that will be used in multiple calculations:

>>> be.configure_lens_set_file('/path/for/current/be_lens_file.npy')

Store sets in the `be_lens_file.npy`:

>>> sets_list_of_tuples = [(3, 0.0001, 1, 0.0002),
                           (1, 0.0001, 1, 0.0003),
                           (2, 0.0001, 1, 0.0005)]
>>> set_lens_set_to_file(sets_list_of_tuples)

Get the first set that was stored in the `be_lens_file.npy`:

>>> be.get_lens_set(1)
(3, 0.0001, 1, 0.0002)

.. note::

   The Be lens holders can take 3 different sets that could be set before experiments so that only the relevant beamline section is vented once. These sets can be stored in a `.npy` file with the :meth:`pcdscalc.be_lens_cals.set_lens_set_to_file` function, and they can be accessed in other calculations with :func:`pcsdcalc.be_lens_cals.get_lens_set`. The `.npy` file containing the lens sets could be then saved to a specific experiment so users know which stack was used for the beamtime.

For more examples please look at each individual function's `Example` section. 

.. currentmodule:: pcdscalc.be_lens_calcs

.. automodule:: pcdscalc.be_lens_calcs
   :members:
   :undoc-members:
   :show-inheritance: