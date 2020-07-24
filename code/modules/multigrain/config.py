"""Configuration."""

import pint

# Required Phantom version.
PHANTOM_VERSION = '666da9e892cb3f2d9f89e132504e185fe2f22f31'

# Path to HDF5 library.
HDF5ROOT = '/usr/local/opt/hdf5'

# Extra compiler arguments.
EXTRA_COMPILER_ARGUMENTS = ['FC=gfortran-9']

# Units.
UNITS = pint.UnitRegistry(system='cgs')
