"""Dusty-wave analysis functions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import exact


def calculate_velocity_density(
    sim, amplitude, sound_speed, num_particles_x=128, width_factor=1.001
):

    snap_initial = sim.snaps[0]
    time = sim.properties['time'].magnitude

    xmin = snap_initial._file_pointer['header/xmin'][()]
    xmax = snap_initial._file_pointer['header/xmax'][()]
    xwidth = xmax - xmin
    dx = xwidth / num_particles_x * width_factor

    subsnaps = [snap_initial['gas']] + snap_initial['dust']
    mean_density_initial = np.array([subsnap['density'].mean() for subsnap in subsnaps])

    n_species = snap_initial.num_dust_species + 1
    density = np.zeros((len(sim), n_species))
    velocity_x = np.zeros((len(sim), n_species))

    for idxi, snap in enumerate(sim.snaps):
        mask = (snap['x'] < xmin + dx) | (snap['x'] > xmax - dx)
        _snap = snap[mask]
        subsnaps = [_snap['gas']] + _snap['dust']
        for idxj, subsnap in enumerate(subsnaps):
            density[idxi, idxj] = (
                subsnap["rho"].mean() - mean_density_initial[idxj]
            ) / (amplitude * mean_density_initial[idxj])
            velocity_x[idxi, idxj] = subsnap["velocity_x"].mean() / (
                amplitude * sound_speed
            )

    arrays = np.hstack((time[:, np.newaxis], density, velocity_x))
    columns = (
        ["time"]
        + [f"density.{idx}" for idx in range(n_species)]
        + [f"velocity_x.{idx}" for idx in range(n_species)]
    )

    return pd.DataFrame(arrays, columns=columns)


def plot_velocity_density(dataframes, figsize=(10, 8)):

    fig, axs = plt.subplots(
        ncols=len(dataframes), nrows=2, sharex=True, figsize=figsize
    )
    fig.subplots_adjust(hspace=0.1)

    for idxi, (name, data) in enumerate(dataframes.items()):
        n_species = len([col for col in data.columns if col.startswith('density')])
        labels = ['Gas'] + [f'Dust {idx}' for idx in range(1, n_species + 1)]
        time = data['time'].to_numpy()
        for idxj in range(n_species):
            ax_v = axs[0, idxi]
            ax_rho = axs[1, idxi]
            v_numerical = data[f'velocity_x.{idxj}'].to_numpy()
            v_exact = exact.velocity_x(time, idxj, n_species)
            rho_numerical = data[f'density.{idxj}'].to_numpy()
            rho_exact = exact.density(time, idxj, n_species)
            [line] = ax_v.plot(
                time, v_numerical, 'o', ms=4, fillstyle='none', label=labels[idxj]
            )
            ax_rho.plot(time, rho_numerical, 'o', ms=4, fillstyle='none')
            ax_v.plot(time, v_exact, color=line.get_color())
            ax_rho.plot(time, rho_exact, color=line.get_color())
        axs[1, idxi].set(xlabel='Time')
    axs[0, 0].set(ylabel='Normalized velocity')
    axs[1, 0].set(ylabel='Normalized density')
    for ax in axs.flatten():
        ax.grid(b=True)

    return fig
