"""Build Phantom for dustyshock and setup calculation."""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
from pathlib import Path

PREFIX = 'dustyshock'

HDF5_DIR = os.getenv('HDF5_DIR')

CODE_DIR = pathlib.Path('~/repos/multigrain/code').expanduser()
IC_DIR = CODE_DIR / 'initial-conditions' / 'dustyshock'
SLURM_FILE = CODE_DIR / 'misc' / 'dustyshock-slurm.swm'

PHANTOM_DIR = pathlib.Path('~/repos/phantom').expanduser()
PHANTOM_VERSION = 'd9a5507f3fd97b5ed5acf4547f82449476b29091'
PHANTOM_PATCHES = [
    CODE_DIR / 'patches' / 'phantom-d9a5507f-multigrain_setup_shock.patch',
    CODE_DIR / 'patches' / 'phantom-d9a5507f-idustbound.patch',
    CODE_DIR / 'patches' / 'phantom-d9a5507f-printing.patch',
]


def main(
    run_name: str,
    run_root_dir: str,
    system: str,
    equation_of_state: str,
    fortran_compiler: str,
    schedule_job: bool,
):
    """Compile Phantom and setup calculation.

    Parameters
    ----------
    run_name
        The name of the run, e.g. 'isothermal'. This corresponds to the
        directory containing the Phantom .setup and .in files.
    run_root_dir
        The path to the directory under which a new run directory will
        be created.
    system
        This is the Phantom Makefile SYSTEM variable. Can be 'ifort' or
        'gfortran'.
    equation_of_state
        Choose the equation of state. Can be 'isothermal' or
        'adiabatic'.
    fortran_compiler
        Use to specify a different fortran compiler version. E.g.
        'gfortran-9'.
    schedule_job
        If True, schedule the run via Slurm.
    """
    if fortran_compiler is None:
        fortran_compiler = system
    data_dir = IC_DIR / run_name
    run_dir = pathlib.Path(run_root_dir).expanduser() / run_name
    if not data_dir.exists():
        raise FileNotFoundError('No initial conditions found for {run_name}')
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
    if equation_of_state == 'isothermal':
        isothermal = 'yes'
    elif equation_of_state == 'adiabatic':
        isothermal = 'no'
    else:
        raise ValueError('equation_of_state must be one of "isothermal" or "adiabatic"')

    # Build Phantom
    _build_phantom(
        isothermal=isothermal, system=system, fortran_compiler=fortran_compiler,
    )

    # Setup calculation
    _setup(run_dir=run_dir, data_dir=data_dir)

    # Schedule calculation
    if schedule_job:
        _schedule(run_dir=run_dir)


def _build_phantom(
    isothermal: str, system: str, fortran_compiler: str = None,
):

    # Get required Phantom version and apply patches
    if not PHANTOM_DIR.exists():
        subprocess.run(
            ['git', 'clone', 'git@bitbucket.org:danielprice/phantom'],
            cwd=PHANTOM_DIR.parent,
            check=True,
        )
    subprocess.run(['git', 'checkout', '--', '*'], cwd=PHANTOM_DIR, check=True)
    subprocess.run(['git', 'checkout', 'master'], cwd=PHANTOM_DIR, check=True)
    subprocess.run(['git', 'pull'], cwd=PHANTOM_DIR, check=True)
    subprocess.run(['git', 'checkout', PHANTOM_VERSION], cwd=PHANTOM_DIR, check=True)
    for patch in PHANTOM_PATCHES:
        subprocess.run(['git', 'apply', patch], cwd=PHANTOM_DIR, check=True)

    # Compile Phantom
    make_options = [
        'SETUP=empty',
        'SETUPFILE=setup_shock.F90',
        'DUST=yes',
        'KERNEL=quintic',
        'PERIODIC=yes',
        f'ISOTHERMAL={isothermal}',
        'MAXP=10000000',
        f'SYSTEM={system}',
        f'FC={fortran_compiler}',
        'HDF5=yes',
        f'HDF5ROOT={HDF5_DIR}',
    ]
    subprocess.run(['make'] + make_options + ['phantom'], cwd=PHANTOM_DIR, check=True)
    subprocess.run(['make'] + make_options + ['setup'], cwd=PHANTOM_DIR, check=True)


def _setup(run_dir: Path, data_dir: Path):

    for file in ['phantom', 'phantomsetup', 'phantom_version']:
        shutil.copy(PHANTOM_DIR / 'bin' / file, run_dir)

    shutil.copy(data_dir / f'{PREFIX}.setup', run_dir)
    shutil.copy(data_dir / f'{PREFIX}.in', run_dir)

    with open(run_dir / f'{PREFIX}00.log', mode='w') as f:
        proc = subprocess.Popen(
            ['./phantomsetup', PREFIX],
            cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)

    shutil.copy(data_dir / f'{PREFIX}.in', run_dir)


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


def _parse_cmdline():

    parser = argparse.ArgumentParser(description='Build Phantom and setup calculation')

    parser.add_argument('run_name', type=str, help="The name of the run, e.g. 'test'.")
    parser.add_argument(
        '--run_root_dir',
        type=str,
        default='.',
        help="The path to where the new run directory will be created.",
    )
    parser.add_argument(
        '--system',
        type=str,
        default='gfortran',
        help="Compiler system Phantom Makefile variable. Can be 'ifort' or 'gfortran'.",
    )
    parser.add_argument(
        '--equation_of_state',
        type=str,
        default='isothermal',
        help="Choose the equation of state: 'isothermal' or 'adiabatic'.",
    )
    parser.add_argument(
        '--fortran_compiler',
        type=str,
        help="Specify a different fortran compiler version. E.g. 'gfortran-9'.",
    )
    parser.add_argument(
        '--schedule_job',
        default=False,
        action='store_true',
        help="Schedule job with Slurm.",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = _parse_cmdline()

    print(f'Run name:                {args.run_name}')
    print(f'Run root dir:            {args.run_root_dir}')
    print(f'Equation of state:       {args.equation_of_state}')
    print(f'SYSTEM:                  {args.system}')
    if args.fortran_compiler is not None:
        print(f'FC:                      {args.fortran_compiler}')
    if args.schedule_job:
        print('\nScheduling job on Slurm after setup')
    print('\n\n\n')

    main(
        run_name=args.run_name,
        run_root_dir=args.run_root_dir,
        system=args.system,
        equation_of_state=args.equation_of_state,
        fortran_compiler=args.fortran_compiler,
        schedule_job=args.schedule_job,
    )
