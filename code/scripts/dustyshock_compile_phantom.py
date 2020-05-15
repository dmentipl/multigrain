"""Build Phantom for dustyshock and setup calculation."""

import shutil
import subprocess
from os import getenv
from pathlib import Path

import click
import phantombuild

PREFIX = 'dustyshock'
HDF5_DIR = getenv('HDF5_DIR')
CODE_DIR = Path('~/repos/multigrain/code').expanduser()
IC_DIR = CODE_DIR / 'initial-conditions' / 'dustyshock'
SLURM_FILE = CODE_DIR / 'misc' / 'dustyshock-slurm.swm'

PHANTOM_DIR = '~/repos/phantom'
PHANTOM_VERSION = 'd9a5507f3fd97b5ed5acf4547f82449476b29091'
PHANTOM_PATCHES = [
    CODE_DIR / 'patches' / 'phantom-d9a5507f-multigrain_setup_shock.patch',
    CODE_DIR / 'patches' / 'phantom-d9a5507f-idustbound.patch',
    CODE_DIR / 'patches' / 'phantom-d9a5507f-printing.patch',
]


@click.command()
@click.option('--run_name', required=True, help='The name of the run.')
@click.option(
    '--root_dir', required=True, help='The directory in which to put run directory.'
)
@click.option('--system', required=True, help='The Phantom SYSTEM Makefile variable.')
@click.option(
    '--equation_of_state',
    required=True,
    help='Choose the equation of state. Can be "isothermal" or "adiabatic".',
)
@click.option('--fortran_compiler', required=False, help='The Fortran compiler.')
@click.option('--schedule_job', required=False, help='Schedule the run via Slurm.')
def main(
    run_name: str,
    root_dir: str,
    system: str,
    equation_of_state: str,
    fortran_compiler: str = None,
    schedule_job: bool = False,
):
    """Compile Phantom and setup calculation."""
    if equation_of_state == 'isothermal':
        isothermal = 'yes'
    elif equation_of_state == 'adiabatic':
        isothermal = 'no'
    else:
        raise ValueError('equation_of_state must be one of "isothermal" or "adiabatic"')

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
    extra_makefile_options = {'ISOTHERMAL': isothermal, 'MAXP': '10000000'}
    if fortran_compiler is not None:
        extra_makefile_options['FC'] = fortran_compiler
    phantombuild.build_phantom(
        phantom_dir=PHANTOM_DIR,
        setup='dustyshock',
        system=system,
        hdf5_location=HDF5_DIR,
        extra_makefile_options=extra_makefile_options,
    )

    # Set up calculation
    run_dir = Path(root_dir).expanduser() / run_name
    input_dir = IC_DIR / run_name
    phantombuild.setup_calculation(
        prefix=PREFIX, run_dir=run_dir, input_dir=input_dir, phantom_dir=PHANTOM_DIR,
    )

    # Schedule calculation
    if schedule_job:
        _schedule(run_dir=run_dir)


def _schedule(run_dir: Path):

    shutil.copy(SLURM_FILE, run_dir)
    try:
        subprocess.run(['sbatch', SLURM_FILE], cwd=run_dir, check=True)
    except FileNotFoundError:
        print(
            '\n\n\nsbatch not available on this machine.\n'
            f'Open tmux, cd to run directory, and type: '
            '"./phantom {PREFIX}.in 2>&1 | tee {PREFIX}01.log"'
        )


if __name__ == "__main__":
    main()
