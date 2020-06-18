"""Radial drift analysis functions."""

import matplotlib.pyplot as plt
import numpy as np
import plonk


def generate_profiles(snap, midplane_height, **kwargs):
    """Generate gas and dust species profiles near midplane."""
    midplane = np.abs(snap['z']) < midplane_height
    profs = [
        plonk.load_profile(subsnap, **kwargs)
        for subsnap in snap[midplane].subsnaps_as_list()
    ]
    return profs


def plot_radial_drift(profs):
    """Plot gas and dust radial drift profiles."""
    time = profs[0].snap.properties['time'].to('year')

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.subplots_adjust(right=0.7)

    for idx, prof in enumerate(profs):
        label = 'gas' if idx == 0 else f'dust {idx}'
        prof.plot('radius', 'radial_velocity_cylindrical', ax=ax, label=label)

    ax.legend(loc='lower left', bbox_to_anchor=(1.01, 0.01), ncol=1)
    ax.grid()
    ax.set_ylim(-0.01, None)
    ax.text(
        0.9,
        0.9,
        f'{time:.0f}',
        ha='right',
        transform=ax.transAxes,
        bbox={'facecolor': 'none'},
    )

    return fig
