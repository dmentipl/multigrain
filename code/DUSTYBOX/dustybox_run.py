"""
Run dustybox calculations.

This script looks in directory for a collection of sub-directories with
Phantom dustybox runs set up with initial conditions. Then it runs each
sequentially.

Daniel Mentiplay, 2019.
"""

import argparse
import pathlib
import shutil
import subprocess
import sys


def main():
    run_root_dir = get_command_line()
    run(run_root_dir)


def run(run_root_dir: pathlib.Path):

    print('\n' + 72 * '-')
    print('>>> Running calculations <<<')
    print(72 * '-' + '\n')

    for directory in sorted(run_root_dir.iterdir()):
        print(f'Running {directory.name}...')
        in_files = list(directory.glob('*.in'))
        if len(in_files) > 1:
            raise ValueError('Too many .in files in directory')
        in_file = in_files[0].name
        log_file = f'{in_files[0].stem}01.log'
        with open(directory / log_file, 'w') as fp:
            subprocess.run(
                [directory / 'phantom', in_file], cwd=directory, stdout=fp, stderr=fp
            )


def get_command_line():
    """
    Get command line options.

    Returns
    -------
    run_root_dir : Path
        The path to the root directory for the calculations.
    """
    parser = argparse.ArgumentParser(description='Set up dustybox calculations')
    parser.add_argument(
        '-r',
        '--run_root_dir',
        help='the root directory for the calculations',
        required=True,
    )
    args = parser.parse_args()
    run_root_dir = pathlib.Path(args.run_root_dir).resolve()
    return run_root_dir


if __name__ == '__main__':
    main()
