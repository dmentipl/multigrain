"""
Run dustybox calculations.

This script looks in directory for a collection of sub-directories with
Phantom dustybox runs set up with initial conditions. Then it runs each
sequentially.

Daniel Mentiplay, 2019.
"""

import pathlib
import shutil
import subprocess
import sys

RUN_ROOT_DIR = pathlib.Path('~/runs/multigrain/dustybox').expanduser()


def main():

    print('\n' + 72 * '-')
    print('>>> Running calculations <<<')
    print(72 * '-' + '\n')

    for directory in sorted(RUN_ROOT_DIR.iterdir()):
        print(f'Running {directory.name}...')
        with open(directory / 'dustybox01.log', 'w') as fp:
            subprocess.run(
                [directory / 'phantom', 'dustybox.in'],
                cwd=directory,
                stdout=fp,
                stderr=fp,
            )


if __name__ == '__main__':
    main()
