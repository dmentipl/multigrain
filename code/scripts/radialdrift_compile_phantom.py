"""Set up radial drift calculations."""

import os
import pathlib
from pathlib import Path

import phantombuild

PREFIX = 'radialdrift'
HDF5_DIR = os.getenv('HDF5_DIR')
CODE_DIR = pathlib.Path('~/repos/multigrain/code').expanduser()

PHANTOM_DIR = pathlib.Path('~/repos/phantom').expanduser()
PHANTOM_VERSION = 'd9a5507f3fd97b5ed5acf4547f82449476b29091'
PHANTOM_PATCHES = [
    CODE_DIR / 'patches' / 'phantom-d9a5507f-multigrain_setup_shock.patch',
    CODE_DIR / 'patches' / 'phantom-d9a5507f-idustbound.patch',
    CODE_DIR / 'patches' / 'phantom-d9a5507f-printing.patch',
]


def main(
    prefix: str,
    run_dir: Path,
    input_dir: Path,
    system: str,
    fortran_compiler: str = None,
):

    # Clone Phantom
    phantombuild.get_phantom(phantom_dir=PHANTOM_DIR)

    # Checkout required version
    phantombuild.checkout_phantom_version(
        phantom_dir=PHANTOM_DIR, required_phantom_git_commit_hash=PHANTOM_VERSION
    )

    # Apply patches
    for patch in PHANTOM_PATCHES:
        phantombuild.patch_phantom(phantom_dir=PHANTOM_DIR, phantom_patch=patch)

    # Compile Phantom
    extra_makefile_options = {'MAXP': '10000000'}
    if fortran_compiler is not None:
        extra_makefile_options['FC'] = fortran_compiler
    phantombuild.build_phantom(
        phantom_dir=PHANTOM_DIR,
        setup='dustydisc',
        system=system,
        hdf5_location=HDF5_DIR,
        extra_makefile_options=extra_makefile_options,
    )

    # Set up calculation
    phantombuild.setup_calculation(
        prefix=prefix, run_dir=run_dir, input_dir=input_dir, phantom_dir=PHANTOM_DIR
    )


if __name__ == '__main__':

    run_dir = pathlib.Path('~/runs/multigrain/radialdrift/test1').expanduser()
    input_dir = CODE_DIR / 'initial-conditions' / 'radialdrift' / 'test1'
    system = 'gfortran'
    fortran_compiler = 'gfortran-9'

    main(
        prefix=PREFIX,
        run_dir=run_dir,
        input_dir=input_dir,
        system=system,
        fortran_compiler=fortran_compiler,
    )
