"""Make animations of dusty-shock simulations."""

from pathlib import Path

import matplotlib.pyplot as plt
import plonk

NAME_GLOB = 'N_*-nx_*-smooth_fac_*-hfact_*'
PATH_1 = Path('/fred/oz015/dmentipl/runs/multigrain/dustyshock')
PATH_3 = Path('/fred/oz015/dmentipl/runs/multigrain/dustyshock2')

YS = ['velocity_x', 'density']
XLIM = (-20, 40)
YLIMS_1 = [(-0.4, 2.8), (-0.4, 10.8)]
YLIMS_3 = [(-0.4, 2.8), (-2.0, 22.0)]


def main():
    loop_over_sims(path=PATH_1, glob=NAME_GLOB, ys=YS, xlim=XLIM, ylims=YLIMS_1)
    loop_over_sims(path=PATH_3, glob=NAME_GLOB, ys=YS, xlim=XLIM, ylims=YLIMS_3)


def loop_over_sims(path, glob, ys, xlim, ylims):
    for p in path.glob(glob):
        print(f'Simulation: {p.name}')
        name = (
            p.name.replace('smooth_fac', 'smooth^fac')
            .replace('_', '=')
            .replace('-', ', ')
            .replace('smooth^fac', 'smooth_fac')
        )
        filename = p / 'anim.mp4'

        print('Loading simulation...')
        sim = plonk.load_sim(prefix='dustyshock', directory=p)

        print('Loading snaps...')
        snaps = sim.snaps

        print('Animating...')
        particle_animation(
            name=name, filename=filename, snaps=snaps, ys=ys, xlim=xlim, ylims=ylims
        )
        print('Finished!')


def particle_animation(name, filename, snaps, ys, xlim, ylims):
    figsize = (9, 3 * len(ys))
    fig, axs = plt.subplots(nrows=len(ys), sharex=True, figsize=figsize)
    fig.subplots_adjust(hspace=0.05)
    marker_style = {'marker': 'o', 'fillstyle': 'none'}

    for y, ylim, ax in zip(ys, ylims, axs):
        plonk.visualize.particle_plot(snap=snaps[0], y=y, **marker_style, ax=ax)
        ax.set(xlim=xlim, ylim=ylim, ylabel=y)

    text = [plonk.utils.time_string(snap, 'second', 's') for snap in snaps]
    axs[0].text(0.9, 0.9, text[0], ha='right', transform=axs[0].transAxes)
    axs[0].set_title(name)
    axs[-1].set_xlabel('x')

    plonk.visualize.animation_particles(
        filename=filename,
        snaps=snaps,
        x='x',
        y=ys,
        fig=fig,
        adaptive_limits=False,
        text=text,
        save_kwargs={'fps': 10, 'dpi': 300},
    )


if __name__ == '__main__':
    main()
