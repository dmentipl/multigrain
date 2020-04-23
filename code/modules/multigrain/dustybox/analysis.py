"""Dusty box analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import exact as exact_solution


def get_dust_properties(snap):
    """Get dust properties.

    Calculate the dust-to-gas ratio and stopping times.

    Parameters
    ----------
    sim
        The Simulation object.

    Returns
    -------
    dust_fraction
        The dust fraction for each dust species.
    stopping_time
        The stopping time for each dust species.
    """
    subsnaps = [snap['gas']] + snap['dust']
    density = np.array([subsnap['density'].mean() for subsnap in subsnaps])
    c_s = np.sqrt(snap.properties['polytropic_constant'])
    ɣ = snap.properties['adiabatic_index']
    s = snap.properties['grain_size'].to(snap.units['length']).magnitude
    rho_m = snap.properties['grain_density'].to(snap.units['density']).magnitude
    rho_g = density[0]
    rho_d = density[1:]
    drag_coeff = rho_g * rho_d * c_s / (np.sqrt(np.pi * ɣ / 8) * s * rho_m)
    dust_fraction = density[1:] / np.sum(density)
    stopping_time = density.sum() / drag_coeff

    return dust_fraction, stopping_time


def calculate_differential_velocity(sim):
    """Calculate differential velocity.

    The results for each simulation is a DataFrame of differential
    velocities for each dust species. The columns are for the
    differential velocity at each time:
        - from the simulation data
        - from the analytic solution with back reaction
        - from the analytic solution without back reaction

    Parameters
    ----------
    sim
        The simulation object.

    Returns
    -------
    The velocity differential DataFrame.
    """
    n_dust = sim.snaps[0].num_dust_species

    # Snapshot times
    time = sim.properties['time'].magnitude

    # Velocity differential: simulation data
    data = np.zeros((len(time), n_dust))
    for idx, snap in enumerate(sim.snaps):
        subsnaps = [snap['gas']] + snap['dust']
        vx = np.array([subsnap['velocity_x'].mean() for subsnap in subsnaps])
        data[idx, :] = vx[1:] - vx[0]

    # Velocity differential: analytical solutions
    dust_fraction, stopping_time = get_dust_properties(sim.snaps[0])
    delta_vx_init = data[0, :]
    exact1 = np.zeros((len(time), n_dust))
    exact2 = np.zeros((len(time), n_dust))
    for idxi, t in enumerate(time):
        exact1[idxi, :] = exact_solution.delta_vx(
            t, stopping_time, dust_fraction, delta_vx_init
        )
        for idxj in range(n_dust):
            exact2[idxi, idxj] = exact_solution.delta_vx(
                t, stopping_time[idxj], dust_fraction[idxj], delta_vx_init[idxj]
            )

    # Generate DataFrame
    arrays = np.hstack([time[:, np.newaxis], data, exact1, exact2])
    columns = (
        ['time']
        + [f'data.{idx}' for idx in range(1, n_dust + 1)]
        + [f'exact1.{idx}' for idx in range(1, n_dust + 1)]
        + [f'exact2.{idx}' for idx in range(1, n_dust + 1)]
    )
    dataframe = pd.DataFrame(arrays, columns=columns)

    return dataframe


def calculate_error(df):
    time = df['time'].to_numpy()
    data = [df[col].to_numpy() for col in df.columns if col.startswith('data')]
    exact = [df[col].to_numpy() for col in df.columns if col.startswith('exact1')]
    error = np.array([np.abs(yd - ye) for yd, ye in zip(data, exact)]).T
    n_dust = len(data)

    # Generate DataFrame
    arrays = np.hstack([time[:, np.newaxis], error])
    columns = ['time'] + [f'error.{idx}' for idx in range(1, n_dust + 1)]
    dataframe = pd.DataFrame(arrays, columns=columns)

    return dataframe


def plot_differential_velocity(df, ax):
    """Plot differential velocity.

    Plot the data as circle markers, the analytical solution with back
    reaction as solid lines, and the analytical solution without back
    reaction as dashed lines.

    Parameters
    ----------
    df
        A DataFrame with the differential velocity.
    ax
        Matplotlib Axes.

    Returns
    -------
    ax
        Matplotlib Axes.
    """
    x = df['time'].to_numpy()
    y_data = [df[col].to_numpy() for col in df.columns if col.startswith('data')]
    y_exact1 = [df[col].to_numpy() for col in df.columns if col.startswith('exact1')]
    y_exact2 = [df[col].to_numpy() for col in df.columns if col.startswith('exact2')]

    for yd, ye1, ye2 in zip(y_data, y_exact1, y_exact2):
        [line] = ax.plot(x, ye1)
        ax.plot(x, ye2, '--', color=line.get_color())
        ax.plot(x, yd, 'o', ms=4, fillstyle='none', color=line.get_color())

    return ax


def plot_differential_velocity_all(dataframes):
    """Plot differential velocity for each simulation.

    Parameters
    ----------
    dataframes
        A dictionary of DataFrames, one per simulation.

    Returns
    -------
    fig
        Matplotlib Figure.
    """
    fig, axs = plt.subplots(ncols=2, nrows=3, sharex=True, sharey=True, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for df, ax in zip(dataframes.values(), axs.T.flatten()):
        plot_differential_velocity(df, ax)
    axs[0, 0].set(title='Dust-to-gas: 0.01')
    axs[0, 1].set(title='Dust-to-gas: 0.5')
    for ax in axs[-1, :]:
        ax.set(xlabel='Time')
    for ax in axs[:, 0]:
        ax.set(ylabel='Differential velocity')
    return fig


def plot_error(df, ax):
    """Plot differential velocity error.

    Parameters
    ----------
    df
        A DataFrame with the differential velocity.
    ax
        Matplotlib Axes.

    Returns
    -------
    ax
        Matplotlib Axes.
    """
    x = df['time'].to_numpy()
    y_error = [df[col].to_numpy() for col in df.columns if col.startswith('error')]

    for y in y_error:
        ax.semilogy(x, y)

    return ax


def plot_error_all(dataframes):
    fig, axs = plt.subplots(ncols=2, nrows=3, sharex=True, sharey=True, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for df, ax in zip(dataframes.values(), axs.T.flatten()):
        plot_error(df, ax)
    axs[0, 0].set(title='Dust-to-gas: 0.01')
    axs[0, 1].set(title='Dust-to-gas: 0.5')
    for ax in axs[-1, :]:
        ax.set(xlabel='Time')
    for ax in axs[:, 0]:
        ax.set(ylabel='Differential velocity error')
    return fig
