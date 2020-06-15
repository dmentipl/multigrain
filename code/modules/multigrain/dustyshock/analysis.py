"""Dusty-shock analysis functions."""

from pathlib import Path

import matplotlib.pyplot as plt
import plonk
import tqdm
from matplotlib import animation

from .exact import velocity as velocity_exact

X_WIDTH_EXACT = 100.0
DUST_TO_GAS = 1.0
DENSITY_LEFT = 1.0
VELOCITY_LEFT = 2.0
MACH_NUMBER = 2.0


def last_snap(directory):
    return sorted(Path(directory).glob('dustyshock_*.h5'))[-1]


def plot_quantity_subsnaps(snap, quantity, ax, xrange):
    mask = (snap['x'] > xrange[0]) & (snap['x'] < xrange[1])
    plonk.visualize.particle_plot(
        snap=snap[mask], x='x', y=quantity, ax=ax, marker='o', fillstyle='none'
    )
    ax.set(xlim=xrange)
    ax.grid()

    return ax


def plot_quantity_profile_subsnaps(snap, quantity, ax, xrange, n_bins):
    subsnaps = snap.subsnaps_as_list()
    for idx, subsnap in enumerate(subsnaps):
        label = 'gas' if idx == 0 else f'dust {idx}'
        prof = plonk.load_profile(
            snap=subsnap,
            radius_min=xrange[0],
            radius_max=xrange[1],
            ndim=1,
            n_bins=n_bins,
        )
        x, y = prof['radius'], prof[quantity]
        ax.plot(x, y, 'o', label=label, fillstyle='none')
    ax.set(xlim=xrange)

    return ax


def make_fig_axs(ncols, width=8, height=4):
    nrows = 2
    fig, axs = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        sharex=True,
        sharey=False,
        squeeze=False,
        figsize=(width, height * nrows),
    )
    return fig, axs


def plot_velocity_density(snaps, xrange, fig_kwargs={}):
    fig, axs = make_fig_axs(ncols=len(snaps), **fig_kwargs)
    for idx, snap in enumerate(snaps):
        plot_quantity_subsnaps(
            snap=snap, quantity='velocity_x', ax=axs[0, idx], xrange=xrange
        )
        plot_quantity_subsnaps(
            snap=snap, quantity='density', ax=axs[1, idx], xrange=xrange
        )
    return fig


def plot_velocity_density_as_profile(snaps, xrange, n_bins=50, fig_kwargs={}):
    fig, axs = make_fig_axs(ncols=len(snaps), **fig_kwargs)
    for idx, snap in enumerate(snaps):
        plot_quantity_profile_subsnaps(
            snap=snap,
            quantity='velocity_x',
            ax=axs[0, idx],
            xrange=xrange,
            n_bins=n_bins,
        )
        plot_quantity_profile_subsnaps(
            snap=snap, quantity='density', ax=axs[1, idx], xrange=xrange, n_bins=n_bins,
        )
    return fig


def plot_velocity_density_exact(drag_coefficients, x_shock, axs):
    n_dust = len(drag_coefficients)
    x, v_gas, v_dust = velocity_exact(
        x_shock=x_shock,
        x_width=X_WIDTH_EXACT,
        n_dust=n_dust,
        dust_to_gas=DUST_TO_GAS,
        drag_coefficient=drag_coefficients,
        density_left=DENSITY_LEFT,
        velocity_left=VELOCITY_LEFT,
        mach_number=MACH_NUMBER,
    )
    p_gas = DENSITY_LEFT * VELOCITY_LEFT / v_gas
    p_dust = DENSITY_LEFT * VELOCITY_LEFT / v_dust

    colors = [line.get_color() for line in axs[0].lines]
    axs[0].plot(x, v_gas, color=colors[0])
    for idx, _v_dust in enumerate(v_dust.T):
        axs[0].plot(x, _v_dust, color=colors[idx + 1])

    colors = [line.get_color() for line in axs[1].lines]
    axs[1].plot(x, p_gas, color=colors[0])
    for idx, _p_dust in enumerate(p_dust.T):
        axs[1].plot(x, _p_dust, color=colors[idx + 1])


def plot_numerical_vs_exact(snaps, xrange, drag_coefficients, x_shock, labels):
    fig = plot_velocity_density(snaps=snaps, xrange=xrange)

    velocity_max = 2.2
    if len(drag_coefficients) == 1:
        density_max = 9.5
    elif len(drag_coefficients) == 3:
        density_max = 18.0
    else:
        raise ValueError('Exact solution must have 1 or 3 dust species')

    label = list(labels.keys())[0]
    _labels = list(labels.values())[0]

    for idx, snap in enumerate(snaps):
        axs = [fig.axes[idx], fig.axes[idx + len(snaps)]]
        plot_velocity_density_exact(
            drag_coefficients=drag_coefficients, x_shock=x_shock[idx], axs=axs
        )
        axs[0].set_ylim(0, velocity_max)
        axs[1].set_ylim(0, density_max)
        axs[0].set_aspect('auto')
        axs[1].set_aspect('auto')

        axs[0].set_title(f'{label}={_labels[idx]}\n time={snap.properties["time"].m}')
        if idx == 0:
            axs[0].set_ylabel('velocity')
            axs[1].set_ylabel('density')


def plot_particle_arrangement(
    *, snap, x='x', y='y', xrange, fig_kwargs={}, plot_kwargs={}
):
    fig, axs = make_fig_axs(ncols=1, **fig_kwargs)
    subsnaps = snap.subsnaps_as_list()
    for idx, (subsnap, ax) in enumerate(zip(subsnaps, axs.flatten())):
        title = 'gas' if idx == 0 else f'dust {idx}'
        _subsnap = subsnap[(subsnap['x'] > xrange[0]) & (subsnap['x'] < xrange[1])]
        plonk.visualize.particle_plot(snap=_subsnap, x=x, y=y, ax=ax, **plot_kwargs)
        ax.set(ylabel=y, title=title)
    return fig


def splash_like_plot(snap, xlim, ylim_density, ylim_velocity_x):

    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(9, 6))
    fig.subplots_adjust(hspace=0.05)

    subsnaps = snap.subsnaps_as_list()

    marker_style = [
        {'linestyle': '', 'marker': 'o', 'markersize': 2, 'fillstyle': 'full'},
        {'linestyle': '', 'marker': 'o', 'markersize': 5, 'fillstyle': 'none'},
        {'linestyle': '', 'marker': 's', 'markersize': 5, 'fillstyle': 'none'},
        {'linestyle': '', 'marker': 'v', 'markersize': 5, 'fillstyle': 'none'},
    ]

    for idx, subsnap in enumerate(subsnaps):
        plonk.visualize.particle_plot(
            snap=subsnap, x='x', y='density', ax=axs[0], **marker_style[idx]
        )
        plonk.visualize.particle_plot(
            snap=subsnap, x='x', y='velocity_x', ax=axs[1], **marker_style[idx]
        )

    axs[0].set(xlim=xlim, ylim=ylim_density, ylabel='density')
    axs[1].set(xlim=xlim, ylim=ylim_velocity_x, xlabel='x', ylabel='x-velocity')
    axs[0].grid()
    axs[1].grid()

    fig.text(0.9, 0.9, f't={snap.properties["time"]}', ha='right')

    return fig, axs


def splash_like_animation(
    snaps, filepath, xlim=(-10, 30), ylim_density=(0, 16), ylim_velocity_x=(0, 2.2)
):
    fig, axs = splash_like_plot(snaps[0], xlim, ylim_density, ylim_velocity_x)

    lines = axs[0].lines + axs[1].lines
    texts = fig.texts

    pbar = tqdm.tqdm(total=len(snaps))

    def animate(idxi):
        pbar.update(n=1)
        time = snaps[idxi].properties['time'].magnitude
        subsnaps = snaps[idxi].subsnaps_as_list()

        texts[0].set_text(f't={time:.0f}')
        for idxj, subsnap in enumerate(subsnaps):
            lines[0 + idxj].set_data(subsnap['x'], subsnap['density'])
            lines[2 + idxj].set_data(subsnap['x'], subsnap['velocity_x'])

        return lines

    anim = animation.FuncAnimation(fig, animate, frames=len(snaps))
    anim.save(filepath, extra_args=['-vcodec', 'libx264'], fps=10, dpi=300)

    pbar.close()

    return anim
