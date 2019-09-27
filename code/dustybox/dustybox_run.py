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
from pathlib import Path


def run_all(run_root_dir: Path):

    print('\n' + 72 * '-')
    print('>>> Running calculations <<<')
    print(72 * '-' + '\n')

    for directory in sorted(run_root_dir.iterdir()):
        if not directory.is_dir():
            continue
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
