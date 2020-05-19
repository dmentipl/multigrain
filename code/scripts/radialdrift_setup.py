"""Set up radial drift calculations."""

from pathlib import Path

import phantombuild

config_file = Path(__file__).parent / 'radialdrift_setup.toml'

runs = phantombuild.read_config(config_file)

for run in runs:
    phantombuild.build_and_setup(**run)
