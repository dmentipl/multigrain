"""Dusty-shock analysis functions."""

import matplotlib.pyplot as plt
import plonk
import tqdm
from matplotlib import animation


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


def splash_like_plot(snap, xlim, ylim_density, ylim_velocity_x):

    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(9, 6))
    fig.subplots_adjust(hspace=0.05)

    subsnaps = [snap['gas']] + snap['dust']

    marker_style = [
        {'linestyle': '', 'marker': 'o', 'markersize': 2, 'fillstyle': 'full'},
        {'linestyle': '', 'marker': 'o', 'markersize': 5, 'fillstyle': 'none'},
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


def splash_like_animation(snaps, filepath):

    xlim = [40, 120]
    ylim_density = [0, 16]
    ylim_velocity_x = [0, 2.2]

    fig, axs = splash_like_plot(snaps[0], xlim, ylim_density, ylim_velocity_x)

    lines = axs[0].lines + axs[1].lines
    texts = fig.texts

    pbar = tqdm.tqdm(total=len(snaps))

    def animate(idxi):
        pbar.update(n=1)
        time = snaps[idxi].properties['time'].magnitude
        subsnaps = [snaps[idxi]['gas']] + snaps[idxi]['dust']

        texts[0].set_text(f't={time:.0f}')
        for idxj, subsnap in enumerate(subsnaps):
            lines[0 + idxj].set_data(subsnap['x'], subsnap['density'])
            lines[2 + idxj].set_data(subsnap['x'], subsnap['velocity_x'])

        return lines

    anim = animation.FuncAnimation(fig, animate, frames=len(snaps))
    anim.save(filepath, extra_args=['-vcodec', 'libx264'], fps=10, dpi=300)

    pbar.close()

    return anim
