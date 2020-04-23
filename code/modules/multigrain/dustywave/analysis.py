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


########################################################################################


def plot_velocity_density(dataframes):

    fig, axs = plt.subplots(ncols=len(dataframes), nrows=2, sharex=True, figsize=(8, 8))

    for idxi, (name, data) in enumerate(dataframes.items()):
        n_species = len([col for col in data.columns if col.startswith('density')])
        time = data['time'].to_numpy()
        for idxj in range(n_species):
            v_numerical = data[f'velocity_x.{idxj}'].to_numpy()
            v_exact = exact.velocity_x(time, idxj, n_species)
            rho_numerical = data[f'density.{idxj}'].to_numpy()
            rho_exact = exact.density(time, idxj, n_species)
            [line] = axs[0, idxi].plot(time, v_numerical, 'o', ms=4, fillstyle='none')
            axs[1, idxi].plot(time, rho_numerical, 'o', ms=4, fillstyle='none')
            axs[0, idxi].plot(time, v_exact, color=line.get_color())
            axs[1, idxi].plot(time, rho_exact, color=line.get_color())

    return fig


########################################################################################


def a():
    t_range = 0.0, 2.0
    rho_range = -1, 1
    vx_range = -1, 1

    ps1 = list()
    ps2 = list()

    colors = Category10[5]

    for run, df in dataframes.items():

        n_species = len([col for col in df.columns if col.startswith('rho')])

        p1 = figure(
            title=f"Normalized density perturbation at x=0 for {run}",
            x_range=t_range,
            y_range=rho_range,
            plot_width=400,
            plot_height=300,
        )
        p2 = figure(
            title=f"Normalized velocity perturbation at x=0 for {run}",
            x_range=t_range,
            y_range=vx_range,
            plot_width=400,
            plot_height=300,
        )

        x = np.array(df["time"])

        for idx in range(n_species):
            legend_label = f"dust {idx}" if idx != 0 else "gas"

            y_numerical = np.array(df[f"rho.{idx}"])
            if idx == 0:
                y_exact = rho_g(time=x, omega=omega[n_species])
            else:
                y_exact = rho_d(
                    time=x, omega=omega[n_species], tstop=tstop[n_species][idx - 1]
                )

            p1.line(x, y_exact, line_color=colors[idx])
            p1.circle(
                x, y_numerical, line_color=colors[idx], fill_color=None
            )  # , legend_label=legend_label

            y_numerical = np.array(df[f"vx.{idx}"])
            if idx == 0:
                y_exact = v_g(time=x, omega=omega[n_species])
            else:
                y_exact = v_d(
                    time=x, omega=omega[n_species], tstop=tstop[n_species][idx - 1]
                )

            p2.line(x, y_exact, line_color=colors[idx])
            p2.circle(
                x, y_numerical, line_color=colors[idx], fill_color=None
            )  # , legend_label=legend_label

        ps1.append(p1)
        ps2.append(p2)

    grid = gridplot([ps2[1], ps2[0], ps1[1], ps1[0]], ncols=2)
    show(grid)
