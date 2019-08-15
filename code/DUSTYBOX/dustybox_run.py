"""
DUSTYBOX run: run Phantom simulations.

Daniel Mentiplay, 2019.
"""

import pathlib
import subprocess
import sys


class PatchError(Exception):
    pass


# ------------------------------------------------------------------------------------ #
# DUSTYBOX parameters
DRAG = 'Kdrag=1.0'
EPS = ['eps1=0.01', 'eps2=0.02', 'eps2=0.03', 'eps2=0.04', 'eps2=0.05']
NDUST = f'ndustlarge={len(EPS)}'

# DUSTYBOX setup and in files
SETUP_FILE = 'dustybox-Kdrag-N=5.setup'
IN_FILE = 'dustybox-Kdrag.in'

# Particular run sub-directory
RUN_SUBDIR = DRAG + '_' + NDUST + '_' + '_'.join(EPS)

# ------------------------------------------------------------------------------------ #
# Directories
RUN_DIR = pathlib.Path('~/runs/multigrain/dustybox').expanduser()
CODE_DIR = pathlib.Path('~/repos/multigrain/code/DUSTYBOX').expanduser()
PHANTOM_DIR = pathlib.Path('~/repos/phantom').expanduser()

# Phantom version
REQUIRED_PHANTOM_GIT_SHA = '6666c55feea1887b2fd8bb87fbe3c2878ba54ed7'
# ------------------------------------------------------------------------------------ #

# Check Phantom version
phantom_git_sha = (
    subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PHANTOM_DIR)
    .strip()
    .decode()
)
if phantom_git_sha != REQUIRED_PHANTOM_GIT_SHA:
    subprocess.check_output(['git', 'checkout', '6666c55f'], cwd=PHANTOM_DIR)

# Apply patch
modified = (
    subprocess.check_output(
        ['git', 'status', '--porcelain', 'setup_dustybox.f90'], cwd=PHANTOM_DIR
    )
    .strip()
    .decode()
)
if modified != '':
    git_patch_output = subprocess.check_output(
        ['git', 'apply', CODE_DIR / 'dustybox.patch'], cwd=PHANTOM_DIR
    )
else:
    diff = (
        subprocess.check_output(
            [
                'diff',
                CODE_DIR / 'setup_dustybox.f90',
                PHANTOM_DIR / 'src/setup/setup_dustybox.f90',
            ]
        )
        .strip()
        .decode()
    )
    if diff != '':
        raise PatchError('Cannot apply patch')

# HDF5 library location
if sys.platform == 'darwin':
    HDF5ROOT = '/usr/local/opt/hdf5'
elif sys.platform == 'linux':
    HDF5ROOT = '/usr/lib/x86_64-linux-gnu/hdf5/serial'
if not pathlib.Path(HDF5ROOT).exists():
    raise FileNotFoundError('Cannot determine HDF5 library location')

# Run directory
run_directory = RUN_DIR / RUN_SUBDIR
run_directory.mkdir()

# Build Phantom
with open(run_directory / 'Makefile', 'w') as fp:
    subprocess.run(
        [PHANTOM_DIR / 'scripts/writemake.sh', 'dustybox'], stdout=fp, stderr=fp
    )
with open(run_directory / 'build-output.log', 'w') as fp:
    subprocess.run(
        [
            'make',
            'SYSTEM=gfortran',
            'HDF5=yes',
            'HDF5ROOT=' + HDF5ROOT,
            'MAXP=10000000',
            'phantom',
            'setup',
        ],
        cwd=run_directory,
        stdout=fp,
        stderr=fp,
    )

# Set up calculation
subprocess.run(['cp', CODE_DIR / SETUP_FILE, run_directory])
with open(run_directory / 'dustybox00.log', 'w') as fp:
    subprocess.run(
        [run_directory / 'phantomsetup', 'dustybox'],
        cwd=run_directory,
        stdout=fp,
        stderr=fp,
    )

# Run calculation
subprocess.run(['cp', CODE_DIR / IN_FILE, run_directory])
with open(run_directory / 'dustybox01.log', 'w') as fp:
    subprocess.run(
        [run_directory / 'phantom', 'dustybox.in'],
        cwd=run_directory,
        stdout=fp,
        stderr=fp,
    )
