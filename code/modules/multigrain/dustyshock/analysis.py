"""Dusty-shock analysis functions."""

import matplotlib.pyplot as plt
import plonk


def plot_quantity_profile_subsnaps(snap, quantity, axs, xrange, yrange):
    subsnaps = [snap['gas']] + snap['dust']
    for idx, (subsnap, ax) in enumerate(zip(subsnaps, axs)):
        label = 'gas' if idx == 0 else f'dust {idx}'
        prof = plonk.load_profile(
            snap=subsnap, radius_min=xrange[0], radius_max=xrange[1], ndim=1
        )
        prof.plot('radius', quantity, label=label, ax=ax, std_dev_shading=True)
        ax.set(xlim=xrange, ylim=yrange, ylabel=quantity)
    return axs


def _make_fig_axs(snap, ncols=2, xwidth=12, ywidth=4):
    nrows = snap.num_dust_species + 1
    fig, axs = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        sharex=True,
        sharey=False,
        figsize=(xwidth, ywidth * nrows),
    )
    return fig, axs


def plot_velocity_density(snap, xrange=(-0.5, 0.5), yrange=None):
    _, axs = _make_fig_axs(snap)
    plot_quantity_profile_subsnaps(
        snap=snap, quantity='velocity_x', axs=axs[:, 0], xrange=xrange, yrange=yrange,
    )
    plot_quantity_profile_subsnaps(
        snap=snap, quantity='density', axs=axs[:, 1], xrange=xrange, yrange=yrange,
    )


def plot_particle_arrangement(snap, x='x', y='y', xrange=(-0.5, 0.5), **kwargs):
    _, axs = _make_fig_axs(snap, ncols=1, xwidth=15, ywidth=3)
    subsnaps = [snap['gas']] + snap['dust']
    for idx, (subsnap, ax) in enumerate(zip(subsnaps, axs)):
        title = 'gas' if idx == 0 else f'dust {idx}'
        _subsnap = subsnap[(subsnap['x'] > xrange[0]) & (subsnap['x'] < xrange[1])]
        plonk.visualize.particle_plot(snap=_subsnap, x=x, y=y, ax=ax, **kwargs)
        ax.set(ylabel=y, title=title)
