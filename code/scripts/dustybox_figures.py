"""Dusty box figures.

Make figures showing time evolution of differential velocity in the
dusty box test:

- for 1, 2, and 5 dust species,
- for dust-to-gas ratio of 0.01 and 0.5.

Also, make figures showing the time evolution of the error of the above
tests.
"""

import pathlib
import sys

import plonk

sys.path.insert(0, str(Path(__file__).resolve().parent / '../modules'))
from multigrain import dustybox


PATH = '~/runs/multigrain/dustybox/time_evolution'

# Path to data
print('Get path to data...')
root_directory = pathlib.Path(PATH).expanduser()
_paths = sorted(list(root_directory.glob('*')))
paths = {p.name: p for p in _paths}

# Calculate velocity differential time evolution
print('Calculate velocity differential time evolution...')
velocity_differential = dict()
for name, path in paths.items():
    print(f'Running analysis for {name}...')
    sim = plonk.load_sim(prefix='dustybox', directory=path)
    velocity_differential[name] = dustybox.calculate_differential_velocity(sim)

# Plot time evolution
print('Plotting figure...')
fig = dustybox.plot_differential_velocity_all(velocity_differential, figsize=(15, 8))
for ax in fig.axes:
    ax.grid()
name = 'dustybox_differential_velocity_comparison.pdf'
print(f'Saving figure to {name}')
fig.savefig(name, bbox_inches='tight', pad_inches=0)

# Calculate velocity differential error
error = dict()
for name, path in paths.items():
    print(f'Calculating error for {name}...')
    error[name] = dustybox.calculate_error(velocity_differential[name])

# Plot error
print('Plotting figure...')
fig = dustybox.plot_error_all(error, figsize=(15, 8))
for ax in fig.axes:
    ax.grid()
name = 'dustybox_differential_velocity_error.pdf'
print(f'Saving figure to {name}')
fig.savefig(name, bbox_inches='tight', pad_inches=0)
