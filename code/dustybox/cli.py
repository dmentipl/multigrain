import pathlib
import sys

import click
import phantombuild

from analysis import do_analysis
from parameters import parameters
from setup import do_setup
from run import do_run

PHANTOM_DIR_DEFAULT = '~/repos/phantom'
PHANTOM_VERSION_DEFAULT = '6666c55feea1887b2fd8bb87fbe3c2878ba54ed7'
if sys.platform == 'darwin':
    HDF5_ROOT_DIR_DEFAULT = '/usr/local/opt/hdf5'
elif sys.platform == 'linux':
    HDF5_ROOT_DIR_DEFAULT = '/usr/lib/x86_64-linux-gnu/hdf5/serial'


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    '-r',
    '--run_root_dir',
    required=True,
    help='the root directory for the calculations',
)
@click.option(
    '--phantom_dir',
    help='the Phantom repository directory',
    default=PHANTOM_DIR_DEFAULT,
)
@click.option(
    '--phantom_version',
    help='the Phantom version specified by git commit hash',
    default=PHANTOM_VERSION_DEFAULT,
)
@click.option(
    '--HDF5_root_dir', help='the path to the HDF5 libary', default=HDF5_ROOT_DIR_DEFAULT
)
def setup(run_root_dir, phantom_dir, phantom_version, HDF5_root_dir):
    run_root_dir = pathlib.Path(run_root_dir).expanduser().resolve()
    phantom_dir = pathlib.Path(phantom_dir).expanduser().resolve()
    HDF5_root_dir = pathlib.Path(HDF5_root_dir).expanduser().resolve()
    phantombuild.get_phantom(phantom_dir=phantom_dir)
    phantombuild.checkout_phantom_version(
        phantom_dir=phantom_dir, required_phantom_git_commit_hash=phantom_version
    )
    do_setup(run_root_dir, parameters, phantom_dir, HDF5_root_dir)


@cli.command()
@click.option(
    '-r',
    '--run_root_dir',
    required=True,
    help='the root directory for the calculations',
)
def run(run_root_dir):
    run_root_dir = pathlib.Path(run_root_dir).expanduser().resolve()
    do_run(run_root_dir)


@cli.command()
@click.option(
    '-r',
    '--run_root_dir',
    required=True,
    help='the root directory for the calculations',
)
@click.option(
    '-f',
    '--force_recompute',
    help='force recomputing of quantities written to .csv files',
)
def analysis(run_root_dir, force_recompute):
    run_root_dir = pathlib.Path(run_root_dir).expanduser().resolve()
    do_analysis(run_root_dir, force_recompute)


if __name__ == '__main__':
    cli()
