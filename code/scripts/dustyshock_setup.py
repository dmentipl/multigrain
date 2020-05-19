"""Set up dusty-shock calculations."""

from pathlib import Path

import phantombuild

config_file = Path(__file__).parent / 'dustyshock_setup.toml'

runs = phantombuild.read_config(config_file)

for run in runs:
    phantombuild.build_and_setup(**run)
