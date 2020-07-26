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
    try:
        snap.physical_units()
    except ValueError:
        pass
    subsnaps = [snap['gas']] + snap['dust']
    density = np.array([subsnap['density'].to('g/cm^3').magnitude.mean() for subsnap in subsnaps])
    c_s = (np.sqrt(snap.properties['polytropic_constant']) * snap.units['velocity']).to('cm/s').magnitude
    ɣ = snap.properties['adiabatic_index']
    s = snap.properties['grain_size'].to('cm').magnitude
    rho_m = snap.properties['grain_density'].to('g/cm^3').magnitude
    rho_g = density[0]
    rho_d = density[1:]
    drag_coeff = rho_g * rho_d * c_s / (np.sqrt(np.pi * ɣ / 8) * s * rho_m)
    dust_fraction = density[1:] / np.sum(density)
    stopping_time = density.sum() / drag_coeff

    return dust_fraction, stopping_time


def calculate_differential_velocity(sim):
    """Calculate differential velocity.

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
        try:
            snap.physical_units()
        except ValueError:
            pass
        subsnaps = [snap['gas']] + snap['dust']
        vx = np.array([subsnap['velocity_x'].to('cm/s').magnitude.mean() for subsnap in subsnaps])
        data[idx, :] = vx[1:] - vx[0]

    # Generate DataFrame
    arrays = np.hstack([time[:, np.newaxis], data])
    columns = ['time'] + [
        f'differential_velocity.{idx}' for idx in range(1, n_dust + 1)
    ]
    dataframe = pd.DataFrame(arrays, columns=columns)

    return dataframe


def calculate_differential_velocity_exact(
    sim, times=None, n_points=1000, backreaction=True
):
    """Calculate differential velocity exact.

    Parameters
    ----------
    sim
        The simulation object.
    times
        Times to evaluate solution.
    n_points
        Default is 1000.
    backreaction
        True or False.

    Returns
    -------
    The velocity differential DataFrame.
    """
    n_dust = sim.snaps[0].num_dust_species

    # Velocity differential: initial data
    snap = sim.snaps[0]
    try:
        snap.physical_units()
    except ValueError:
        pass
    subsnaps = [snap['gas']] + snap['dust']
    vx = np.array([subsnap['velocity_x'].to('cm/s').magnitude.mean() for subsnap in subsnaps])
    delta_vx_init = vx[1:] - vx[0]

    # Time
    if times is None:
        _time = sim.properties['time'].magnitude
        time = np.linspace(_time[0], _time[-1], n_points)
    else:
        time = times

    # Velocity differential: analytical solutions
    dust_fraction, stopping_time = get_dust_properties(sim.snaps[0])
    exact = np.zeros((len(time), n_dust))
    if backreaction:
        for idxi, t in enumerate(time):
            exact[idxi, :] = exact_solution.delta_vx(
                t, stopping_time, dust_fraction, delta_vx_init
            )
    else:
        for idxi, t in enumerate(time):
            for idxj in range(n_dust):
                exact[idxi, idxj] = exact_solution.delta_vx(
                    t, stopping_time[idxj], dust_fraction[idxj], delta_vx_init[idxj]
                )

    # Generate DataFrame
    arrays = np.hstack([time[:, np.newaxis], exact])
    columns = ['time'] + [
        f'differential_velocity.{idx}' for idx in range(1, n_dust + 1)
    ]
    dataframe = pd.DataFrame(arrays, columns=columns)

    return dataframe


def calculate_error(sim):
    _data = calculate_differential_velocity(sim)
    time = _data['time'].to_numpy()
    data = [_data[col].to_numpy() for col in _data.columns if col.startswith('d')]
    _exact = calculate_differential_velocity_exact(sim, times=time)
    exact = [_exact[col].to_numpy() for col in _exact.columns if col.startswith('d')]
    error = np.array([np.abs(yd - ye) for yd, ye in zip(data, exact)]).T
    n_dust = len(data)

    # Generate DataFrame
    arrays = np.hstack([time[:, np.newaxis], error])
    columns = ['time'] + [f'error.{idx}' for idx in range(1, n_dust + 1)]
    dataframe = pd.DataFrame(arrays, columns=columns)

    return dataframe


def plot_differential_velocity(data, exact1, exact2, ax):
    """Plot differential velocity.

    Plot the data as circle markers, the analytical solution with back
    reaction as solid lines, and the analytical solution without back
    reaction as dashed lines.

    Parameters
    ----------
    data
        A DataFrame with the differential velocity.
    exact1
        A DataFrame with the differential velocity exact solution with
        backreaction.
    exact2
        A DataFrame with the differential velocity exact solution
        without backreaction.
    ax
        Matplotlib Axes.

    Returns
    -------
    ax
        Matplotlib Axes.
    """
    y_data = [data[col].to_numpy() for col in data.columns if col.startswith('d')]
    y_exact1 = [exact1[col].to_numpy() for col in exact1.columns if col.startswith('d')]
    y_exact2 = [exact2[col].to_numpy() for col in exact2.columns if col.startswith('d')]

    for yd, ye1, ye2 in zip(y_data, y_exact1, y_exact2):
        [line] = ax.plot(exact1['time'], ye1)
        ax.plot(exact2['time'], ye2, '--', color=line.get_color(), alpha=0.33)
        ax.plot(data['time'], yd, 'o', ms=4, fillstyle='none', color=line.get_color())

    ax.grid()

    return ax


def plot_differential_velocity_all(
    data, exact1, exact2, ncols=3, figsize=(15, 8), transpose=False
):
    """Plot differential velocity for each simulation.

    Parameters
    ----------
    data
        A dictionary of DataFrames with the differential velocity.
    exact1
        A dictionary of DataFrames with the differential velocity exact
        solution with backreaction.
    exact2
        A dictionary of DataFrames with the differential velocity exact
        solution without backreaction.
    ncols
        The number of columns of axes in the figure. Default is 3.
    figsize
        The figsize like (x, y). Default is (15, 8).
    transpose
        Whether to run along columns or rows. Default (False) is to run
        along rows.

    Returns
    -------
    fig
        Matplotlib Figure.
    """
    nrows = int(np.ceil(len(data) / ncols))
    fig, axs = plt.subplots(
        ncols=ncols, nrows=nrows, sharex=True, sharey=True, figsize=figsize
    )
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    if transpose:
        _axs = axs.T.flatten()
    else:
        _axs = axs.flatten()
    for d, e1, e2, ax in zip(data.values(), exact1.values(), exact2.values(), _axs):
        plot_differential_velocity(d, e1, e2, ax)
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

    ax.grid()

    return ax


def plot_error_all(dataframes, ncols=3, figsize=(15, 8), transpose=False):
    """Plot differential velocity error for each simulation.

    Parameters
    ----------
    dataframes
        A dictionary of DataFrames, one per simulation.
    ncols
        The number of columns of axes in the figure. Default is 3.
    figsize
        The figsize like (x, y). Default is (15, 8).
    transpose
        Whether to run along columns or rows. Default (False) is to run
        along rows.

    Returns
    -------
    fig
        Matplotlib Figure.
    """
    nrows = int(np.ceil(len(dataframes) / ncols))
    fig, axs = plt.subplots(
        ncols=ncols, nrows=nrows, sharex=True, sharey=True, figsize=figsize
    )
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    if transpose:
        _axs = axs.T.flatten()
    else:
        _axs = axs.flatten()
    for df, ax in zip(dataframes.values(), _axs):
        plot_error(df, ax)
    for ax in axs[-1, :]:
        ax.set(xlabel='Time')
    for ax in axs[:, 0]:
        ax.set(ylabel='Differential velocity error')
    return fig
