"""
DUSTYBOX run: run Phantom simulations.

Daniel Mentiplay, 2019.
"""

import pathlib
import subprocess
import sys

# ------------------------------------------------------------------------------------ #
# Set parameters
DRAG = 'Kdrag=1.0'
NDUST = 'ndustlarge=2'
EPS = ['eps1=0.01', 'eps2=0.02']
# ------------------------------------------------------------------------------------ #

# Check Phantom version
PHANTOM_DIR = pathlib.Path('~/repos/phantom').expanduser()
PHANTOM_GIT_SHA = '6666c55feea1887b2fd8bb87fbe3c2878ba54ed7'
phantom_git_sha = (
    subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PHANTOM_DIR)
    .strip()
    .decode()
)
if phantom_git_sha != PHANTOM_GIT_SHA:
    subprocess.check_output(['git', 'checkout', '6666c55f'], cwd=PHANTOM_DIR)

# Apply patch
git_patch_output = subprocess.check_output(
    [
        'git',
        'apply',
        pathlib.Path('~/repos/multigrain/code/DUSTYBOX/dustybox.patch').expanduser(),
    ],
    cwd=PHANTOM_DIR,
)

# HDF5 library location
if sys.platform == 'darwin':
    HDF5ROOT = '/usr/local/opt/hdf5'
elif sys.platform == 'linux':
    HDF5ROOT = '/usr/lib/x86_64-linux-gnu/hdf5/serial'
if not pathlib.Path(HDF5ROOT).exists():
    raise FileNotFoundError('Cannot determine HDF5 library location')

# Setup and in file directory
INPUT_DIRECTORY = pathlib.Path('~/repos/multigrain/code/DUSTYBOX').expanduser()

# Run directory
run_directory = pathlib.Path('~/runs/multigrain/dustybox/').expanduser() / (
    DRAG + '_' + NDUST + '_' + '_'.join(EPS)
)
run_directory.mkdir()

# Build Phantom
with open(run_directory / 'Makefile', 'w') as fp:
    subprocess.run(
        [pathlib.Path('~/repos/phantom/scripts/writemake.sh').expanduser(), 'dustybox'],
        stdout=fp,
        stderr=fp,
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
subprocess.run(['cp', INPUT_DIRECTORY / 'dustybox.setup', run_directory])
with open(run_directory / 'dustybox00.log', 'w') as fp:
    subprocess.run(
        [run_directory / 'phantomsetup', 'dustybox'],
        cwd=run_directory,
        stdout=fp,
        stderr=fp,
    )

# Run calculation
subprocess.run(['cp', INPUT_DIRECTORY / 'dustybox.in', run_directory])
with open(run_directory / 'dustybox01.log', 'w') as fp:
    subprocess.run(
        [run_directory / 'phantom', 'dustybox.in'],
        cwd=run_directory,
        stdout=fp,
        stderr=fp,
    )
