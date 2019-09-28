"""
Run dustybox calculations.

The function 'do_run' looks in directory for a collection of
sub-directories with Phantom dustybox runs set up with initial
conditions. Then it runs each sequentially.
"""

import subprocess
from pathlib import Path
from typing import List


def do_run(run_root_dir: Path) -> List[subprocess.CompletedProcess]:

    print('\n' + 72 * '-')
    print('>>> Running calculations <<<')
    print(72 * '-' + '\n')

    results = list()
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
            result = subprocess.run(
                [directory / 'phantom', in_file], cwd=directory, stdout=fp, stderr=fp
            )
        results.append(result)

    return results