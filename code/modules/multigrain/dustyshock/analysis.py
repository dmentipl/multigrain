"""Dusty-shock analysis functions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plonk
import tqdm
from matplotlib import animation
from scipy.optimize import minimize_scalar

from .exact import velocity as velocity_exact
from .exact import density as density_exact

X_WIDTH_EXACT = 100.0
DUST_TO_GAS = 1.0
DENSITY_LEFT = 1.0
VELOCITY_LEFT = 2.0
MACH_NUMBER = 2.0


def first_snap(directory):
    return plonk.load_snap(sorted(Path(directory).glob('dustyshock_*.h5'))[0])


def last_snap(directory):
    return plonk.load_snap(sorted(Path(directory).glob('dustyshock_*.h5'))[-1])


def find_x_shock(snap, drag_coefficients, xrange):
    print('Finding x-shock via optimization, may take some time...')

    def fn(x_shock):
        return velocity_error_norm(
            snap=snap,
            drag_coefficients=drag_coefficients,
            x_shock=x_shock,
            xrange=xrange,
        )

    result = minimize_scalar(fn, bounds=xrange)
    if result.success:
        return result.x
    print('Failed to find x-shock')
    return


def plot_quantity_subsnaps(snap, quantity, ax, xrange):
    mask = (snap['x'] > xrange[0]) & (snap['x'] < xrange[1])
    plonk.visualize.particle_plot(
        snap=snap[mask], x='x', y=quantity, ax=ax, marker='o', fillstyle='none'
    )
    ax.set(xlim=xrange)
    ax.grid()

    return ax


def plot_quantity_profile_subsnaps(snap, quantity, ax, xrange, n_bins):
    subsnaps = [snap['gas']] + snap['dust']
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


def plot_velocity_density_as_particles(snaps, xrange, fig_kwargs={}):
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


def velocity_error(
    snap, drag_coefficients, x_shock, xrange, error_type='absolute', n_bins=50
):
    if error_type not in ['absolute', 'relative']:
        raise ValueError('Wrong error type: must be "absolute" or "relative"')
    n_dust = len(drag_coefficients)
    v_gas, v_dusts = velocity_exact(
        x_shock=x_shock,
        x_width=X_WIDTH_EXACT,
        n_dust=n_dust,
        dust_to_gas=DUST_TO_GAS,
        drag_coefficient=drag_coefficients,
        density_left=DENSITY_LEFT,
        velocity_left=VELOCITY_LEFT,
        mach_number=MACH_NUMBER,
    )
    v_exact = [v_gas] + v_dusts

    subsnaps = [snap['gas']] + snap['dust']
    error = list()
    for idx, subsnap in enumerate(subsnaps):
        prof = plonk.load_profile(
            snap=subsnap,
            radius_min=xrange[0],
            radius_max=xrange[1],
            ndim=1,
            n_bins=n_bins,
        )
        x, v_numerical = prof['radius'], prof['velocity_x']
        if error_type == 'absolute':
            error.append(np.abs(v_exact[idx](x) - v_numerical))
        elif error_type == 'relative':
            error.append(np.abs((v_exact[idx](x) - v_numerical) / v_exact[idx](x)))

    return x, error[0], error[1:]


def density_error(
    snap, drag_coefficients, x_shock, xrange, error_type='absolute', n_bins=50
):
    if error_type not in ['absolute', 'relative']:
        raise ValueError('Wrong error type: must be "absolute" or "relative"')
    n_dust = len(drag_coefficients)
    d_gas, d_dusts = density_exact(
        x_shock=x_shock,
        x_width=X_WIDTH_EXACT,
        n_dust=n_dust,
        dust_to_gas=DUST_TO_GAS,
        drag_coefficient=drag_coefficients,
        density_left=DENSITY_LEFT,
        velocity_left=VELOCITY_LEFT,
        mach_number=MACH_NUMBER,
    )
    d_exact = [d_gas] + d_dusts

    subsnaps = [snap['gas']] + snap['dust']
    error = list()
    for idx, subsnap in enumerate(subsnaps):
        prof = plonk.load_profile(
            snap=subsnap,
            radius_min=xrange[0],
            radius_max=xrange[1],
            ndim=1,
            n_bins=n_bins,
        )
        x, d_numerical = prof['radius'], prof['density']
        if error_type == 'absolute':
            error.append(np.abs(d_exact[idx](x) - d_numerical))
        elif error_type == 'relative':
            error.append(np.abs((d_exact[idx](x) - d_numerical) / d_exact[idx](x)))

    return x, error[0], error[1:]


def velocity_error_norm(snap, drag_coefficients, x_shock, xrange, n_bins=50):
    n_dust = len(drag_coefficients)
    v_gas, v_dusts = velocity_exact(
        x_shock=x_shock,
        x_width=X_WIDTH_EXACT,
        n_dust=n_dust,
        dust_to_gas=DUST_TO_GAS,
        drag_coefficient=drag_coefficients,
        density_left=DENSITY_LEFT,
        velocity_left=VELOCITY_LEFT,
        mach_number=MACH_NUMBER,
    )
    v_exact = [v_gas] + v_dusts

    subsnaps = [snap['gas']] + snap['dust']
    error_squared = 0.0
    for idx, subsnap in enumerate(subsnaps):
        prof = plonk.load_profile(
            snap=subsnap,
            radius_min=xrange[0],
            radius_max=xrange[1],
            ndim=1,
            n_bins=n_bins,
        )
        x, v_numerical = prof['radius'], prof['velocity_x']
        error_squared += np.sum((v_exact[idx](x) - v_numerical) ** 2)

    return np.sqrt(error_squared)


def density_error_norm(snap, drag_coefficients, x_shock, xrange, n_bins=50):
    n_dust = len(drag_coefficients)
    d_gas, d_dusts = density_exact(
        x_shock=x_shock,
        x_width=X_WIDTH_EXACT,
        n_dust=n_dust,
        dust_to_gas=DUST_TO_GAS,
        drag_coefficient=drag_coefficients,
        density_left=DENSITY_LEFT,
        velocity_left=VELOCITY_LEFT,
        mach_number=MACH_NUMBER,
    )
    d_exact = [d_gas] + d_dusts

    subsnaps = [snap['gas']] + snap['dust']
    error_squared = 0.0
    for idx, subsnap in enumerate(subsnaps):
        prof = plonk.load_profile(
            snap=subsnap,
            radius_min=xrange[0],
            radius_max=xrange[1],
            ndim=1,
            n_bins=n_bins,
        )
        x, d_numerical = prof['radius'], prof['velocity_x']
        error_squared += np.sum((d_exact[idx](x) - d_numerical) ** 2)

    return np.sqrt(error_squared)


def plot_velocity_error_convergence(
    snaps, nxs, drag_coefficients, x_shock, xrange, n_bins=50
):
    err = list()
    for snap, _x_shock in zip(snaps, x_shock):
        err.append(
            velocity_error_norm(
                snap=snap,
                drag_coefficients=drag_coefficients,
                x_shock=_x_shock,
                xrange=xrange,
                n_bins=n_bins,
            )
        )
    fig, ax = plt.subplots()
    ax.plot(np.log10(nxs), np.log10(err))
    ax.set_xlabel('log10(Resolution [nx])')
    ax.set_ylabel('log10(Error [L2 norm])')


def plot_velocity_density_error(
    snap, drag_coefficients, x_shock, xrange, error_type='absolute', n_points=1000
):
    fig, axs = make_fig_axs(ncols=1, width=8, height=4)
    axs = axs.flatten()

    x, verr_gas, verr_dusts = velocity_error(
        snap=snap,
        drag_coefficients=drag_coefficients,
        x_shock=x_shock,
        xrange=xrange,
        error_type=error_type,
    )
    x, derr_gas, derr_dusts = density_error(
        snap=snap,
        drag_coefficients=drag_coefficients,
        x_shock=x_shock,
        xrange=xrange,
        error_type=error_type,
    )

    axs[0].plot(x, verr_gas)
    for verr_dust in verr_dusts:
        axs[0].plot(x, verr_dust)
    axs[0].set_ylabel(f'Velocity {error_type} error')

    axs[1].plot(x, derr_gas)
    for derr_dust in derr_dusts:
        axs[1].plot(x, derr_dust)
    axs[1].set_ylabel(f'Density {error_type} error')
    axs[1].set_xlabel('x')

    for ax in axs:
        ax.grid()

    return fig, axs


def plot_velocity_density_exact(drag_coefficients, x_shock, axs, n_points=1000):
    n_dust = len(drag_coefficients)
    x = np.linspace(x_shock - X_WIDTH_EXACT / 2, x_shock + X_WIDTH_EXACT / 2, n_points)
    _v_gas, _v_dusts = velocity_exact(
        x_shock=x_shock,
        x_width=X_WIDTH_EXACT,
        n_dust=n_dust,
        dust_to_gas=DUST_TO_GAS,
        drag_coefficient=drag_coefficients,
        density_left=DENSITY_LEFT,
        velocity_left=VELOCITY_LEFT,
        mach_number=MACH_NUMBER,
    )

    v_gas = _v_gas(x)
    v_dusts = [_v_dust(x) for _v_dust in _v_dusts]
    rho_gas = DENSITY_LEFT * VELOCITY_LEFT / v_gas
    rho_dusts = [DENSITY_LEFT * VELOCITY_LEFT / v_dust for v_dust in v_dusts]

    colors = [line.get_color() for line in axs[0].lines]
    axs[0].plot(x, v_gas, color=colors[0])
    for idx, v_dust in enumerate(v_dusts):
        axs[0].plot(x, v_dust, color=colors[idx + 1])

    colors = [line.get_color() for line in axs[1].lines]
    axs[1].plot(x, rho_gas, color=colors[0])
    for idx, rho_dust in enumerate(rho_dusts):
        axs[1].plot(x, rho_dust, color=colors[idx + 1])


def plot_numerical_vs_exact(
    snaps,
    xrange,
    drag_coefficients,
    x_shock,
    labels=None,
    plot_type='particles',
    fig_kwargs={},
    n_bins=50,
):
    if plot_type == 'particles':
        fig = plot_velocity_density_as_particles(
            snaps=snaps, xrange=xrange, fig_kwargs=fig_kwargs
        )
    elif plot_type == 'profile':
        fig = plot_velocity_density_as_profile(
            snaps=snaps, xrange=xrange, n_bins=n_bins, fig_kwargs=fig_kwargs
        )
    else:
        raise ValueError('plot_type must be "particles" or "profile"')

    if not isinstance(drag_coefficients[0], list):
        _drag_coefficients = [drag_coefficients for _ in snaps]
    else:
        _drag_coefficients = drag_coefficients

    velocity_max = 2.2

    if labels is not None:
        label = list(labels.keys())[0]
        _labels = list(labels.values())[0]

    for idx, snap in enumerate(snaps):
        K = _drag_coefficients[idx]
        if len(K) == 1:
            density_max = 9.5
        elif len(K) == 3:
            density_max = 18.0
        else:
            density_max = None
        axs = [fig.axes[idx], fig.axes[idx + len(snaps)]]
        plot_velocity_density_exact(drag_coefficients=K, x_shock=x_shock[idx], axs=axs)
        axs[0].set_ylim(0, velocity_max)
        axs[1].set_ylim(0, density_max)
        axs[0].set_aspect('auto')
        axs[1].set_aspect('auto')

        if labels is not None:
            axs[0].set_title(
                f'{label}={_labels[idx]}\n time={snap.properties["time"].m}'
            )
        axs[1].set_xlabel('x')
        if idx == 0:
            axs[0].set_ylabel('Velocity')
            axs[1].set_ylabel('Density')

    return fig


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


def particle_animation(filename, snaps, xlim=None, dlim=None, vlim=None):

    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(9, 6))
    fig.subplots_adjust(hspace=0.05)
    marker_style = {'marker': 'o', 'fillstyle': 'none'}

    plonk.visualize.particle_plot(
        snap=snaps[0], y='velocity_x', **marker_style, ax=axs[0]
    )
    plonk.visualize.particle_plot(snap=snaps[0], y='density', **marker_style, ax=axs[1])

    axs[0].set(xlim=xlim, ylim=vlim, ylabel='velocity')
    axs[1].set(xlim=xlim, ylim=dlim, ylabel='density', xlabel='x')

    plonk.visualize.animation_particles(
        filename=filename,
        snaps=snaps,
        x='x',
        y=['velocity_x', 'density'],
        fig=fig,
        adaptive_limits=False,
        save_kwargs={'fps': 10, 'dpi': 300},
    )


def image_animation(filename, snaps, extent):

    interp = 'cross_section'

    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(9, 2))
    fig.subplots_adjust(hspace=0.05)

    plonk.visualize.plot(
        snap=snaps[0], quantity='velocity_x', interp=interp, ax=axs[0], extent=extent
    )
    plonk.visualize.plot(
        snap=snaps[0], quantity='density', interp=interp, ax=axs[1], extent=extent
    )
    axs[0].set_aspect('auto')
    axs[1].set_aspect('auto')

    plonk.visualize.animation(
        filename=filename,
        snaps=snaps,
        quantity=['velocity_x', 'density'],
        fig=fig,
        interp=interp,
        extent=extent,
        save_kwargs={'fps': 10, 'dpi': 300},
    )
