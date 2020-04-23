"""Dusty-shock analysis functions."""

import matplotlib.pyplot as plt
import plonk


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
        prof.plot('radius', quantity, label=label, ax=ax, std_dev_shading=True)
        ax.set(xlim=xrange)
    return ax


def make_fig_axs(ncols, width=8, height=4):
    nrows = 2
    fig, axs = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        sharex=True,
        sharey=False,
        figsize=(width, height * nrows),
    )
    return fig, axs


def plot_velocity_density(snaps, xrange, n_bins=50, fig_kwargs={}):
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
            snap=snap, quantity='density', ax=axs[1, idx], xrange=xrange, n_bins=n_bins
        )
    return fig


def plot_particle_arrangement(
    *, snap, x='x', y='y', xrange, fig_kwargs={}, plot_kwargs={}
):
    fig, axs = make_fig_axs(snap=snap, ncols=1, **fig_kwargs)
    subsnaps = [snap['gas']] + snap['dust']
    for idx, (subsnap, ax) in enumerate(zip(subsnaps, axs)):
        title = 'gas' if idx == 0 else f'dust {idx}'
        _subsnap = subsnap[(subsnap['x'] > xrange[0]) & (subsnap['x'] < xrange[1])]
        plonk.visualize.particle_plot(snap=_subsnap, x=x, y=y, ax=ax, **plot_kwargs)
        ax.set(ylabel=y, title=title)
    return fig
