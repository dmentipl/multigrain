"""Test problems for multigrain paper."""

# Required Phantom version.
PHANTOM_VERSION = 'e0d6986df99d980267f712d2218376dd50701117'

# Path to HDF5 library.
HDF5ROOT = '/usr/local/opt/hdf5'

# Extra compiler arguments.
EXTRA_COMPILER_ARGUMENTS = ['FC=gfortran-9']

from . import dustybox, dustywave, dustyshock
from .run import setup_multiple_calculations, run_multiple_calculations