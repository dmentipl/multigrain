"""Dusty wave figures.

Make figures showing time evolution of the velocity and density in the
dusty wave test:

- for 1 and 4 dust species.
"""

import pathlib
import sys

import plonk

sys.path.insert(0, '../modules')
from multigrain import dustywave


PATH = '~/runs/multigrain/dustywave'

# Path to data
print('Get path to data...')
root_directory = pathlib.Path(PATH).expanduser()
_paths = sorted(list(root_directory.glob('*')))
paths = {p.name: p for p in _paths}

# Set the sound speed and wave amplitude
sound_speed = 1.0
amplitude = 1e-4

# Set the number of particles in the x direction.
num_particles_x = 128

# Calculate velocity and density time evolution
print('Calculate velocity and density time evolution...')
dataframes = dict()
for name, path in paths.items():
    print(f'Running analysis for {name}...')
    sim = plonk.load_sim(prefix='dustywave', directory=path)
    dataframes[name] = dustywave.calculate_velocity_density(
        sim, amplitude, sound_speed, num_particles_x
    )

# Plot results
print('Plotting figure...')
fig = dustywave.plot_velocity_density(dataframes, figsize=(10, 8))
for ax in fig.axes:
    ax.grid()
name = 'dustywave_velocity_density.pdf'
print(f'Saving figure to {name}')
fig.savefig(name, bbox_inches='tight', pad_inches=0)
