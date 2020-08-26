"""Radial drift velocity.

See the following reference for more details.

- Dipierro et al., 2018, MNRAS, 479, 3, p.4187-4206
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import plonk
from plonk import Profile
from plonk import Snap
from plonk._units import Quantity


PARAMETERS = {
    'dirname': '~/runs/multigrain/radialdrift/test1',
    'filename': 'radialdrift_00200.h5',
    'debug': False,
    'dust_species_to_plot': [0, 1, 2, 3, 4],
    'radius_min': 25 * plonk.units.au,
    'radius_max': 140 * plonk.units.au,
    'scale_height_fac': 0.05,
    'n_bins': 25,
}


def main(parameters):

    dirname = parameters['dirname']
    filename = parameters['filename']
    debug = parameters['debug']
    dust_species_to_plot = parameters['dust_species_to_plot']
    radius_min = parameters['radius_min']
    radius_max = parameters['radius_max']
    scale_height_fac = parameters['scale_height_fac']
    n_bins = parameters['n_bins']

    path = Path(dirname).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f'{dirname} does not exist')
    in_file = path / 'radialdrift.in'
    if not in_file.exists():
        raise FileNotFoundError('{dirname}/radialdrift.in does not exist')
    snap_file = path / filename
    if not snap_file.exists():
        raise FileNotFoundError('{dirname}/{filename} does not exist')

    print(f'          file name = {dirname}/{filename}')
    print(f'         radius min = {radius_min:~}')
    print(f'         radius max = {radius_max:~}')
    print(f'scale height factor = {scale_height_fac}')
    print(f'             n_bins = {n_bins}')

    snap = plonk.load_snap(filename=snap_file)

    profs = calculate_profiles(
        snap=snap,
        radius_min=radius_min,
        radius_max=radius_max,
        scale_height_fac=scale_height_fac,
        n_bins=n_bins,
    )

    ax = plot_profiles(
        snap=snap, profs=profs, dust_species_to_plot=dust_species_to_plot, debug=debug
    )

    plt.show()

    return snap, profs, ax


def calculate_profiles(
    snap: Snap,
    radius_min: Quantity,
    radius_max: Quantity,
    scale_height_fac: float,
    n_bins: int = 50,
) -> Dict[str, List[Profile]]:
    """Calculate radial drift velocity profiles.

    Parameters
    ----------
    snap
        The Snap object.
    radius_min
        The minimum radius for the profiles.
    radius_max
        The maximum radius for the profiles.
    scale_height_fac
        A factor of the scale height within which to average over.
    n_bins
        The number of bins in the profile. Default is 50.

    Returns
    -------
    Dict
        A dictionary of list of profiles. The keys are 'gas' and 'dust'
        and the values are lists of profiles, one per sub-type.
    """
    print('Calculating profiles...')

    snap.add_quantities('disc')
    snap.set_gravitational_parameter(0)
    gamma = snap.properties['adiabatic_index']
    num_dust = snap.num_dust_species

    # Use particles in the midplane only
    # Choose particles such that they are within a factor of the gas scale height
    gas = snap.family('gas')
    prof = plonk.load_profile(snap=gas, cmin=radius_min, cmax=radius_max)
    scale_height = prof.to_function('scale_height')
    snap_midplane = snap[np.abs(snap['z']) < scale_height_fac * scale_height(snap['R'])]

    subsnaps = snap_midplane.subsnaps_as_dict()

    # Create radial profiles for the gas and each dust species
    cmin, cmax = radius_min, radius_max
    profs: Dict[str, List[Profile]] = {'gas': list(), 'dust': list()}
    profs['gas'] = [
        plonk.load_profile(subsnaps['gas'], cmin=cmin, cmax=cmax, n_bins=n_bins)
    ]
    for subsnap in subsnaps['dust']:
        profs['dust'].append(
            plonk.load_profile(subsnap, cmin=cmin, cmax=cmax, n_bins=n_bins)
        )

    p = profs['gas'][0]

    # velocity_pressure is (15) in Dipierro+2018
    p['velocity_pressure'] = np.gradient(p['pressure'], p['radius']) / (
        p['density'] * p['keplerian_frequency']
    )

    # shear_viscosity is between (16) an (17) in Dipierro+2018
    p['shear_viscosity'] = p['disc_viscosity'] * p['density']

    # velocity_visc is (16) in Dipierro+2018
    p['velocity_visc'] = np.gradient(
        p['shear_viscosity']
        * p['radius'] ** 3
        * np.gradient(p['keplerian_frequency'], p['radius']),
        p['radius'],
    ) / (
        p['radius']
        * p['density']
        * np.gradient(p['radius'] ** 2 * p['keplerian_frequency'], p['radius'])
    )

    for idx, prof_dust in enumerate(profs['dust']):
        p[f'midplane_dust_to_gas_{idx+1:03}'] = prof_dust['density'] / p['density']
        p[f'_midplane_stokes_number_{idx+1:03}'] = (
            np.sqrt(np.pi * gamma / 8)
            * snap.properties['grain_density'][idx]
            * snap.properties['grain_size'][idx]
            * p['keplerian_frequency']
            / (p['density'] * p['sound_speed'])
        )

    # lambda_0 and lambda_1 are (17) in Dipierro+2018
    l0 = np.zeros(len(p)) * plonk.units['dimensionless']
    l1 = np.zeros(len(p)) * plonk.units['dimensionless']
    for idx in range(num_dust):
        St = p[f'_midplane_stokes_number_{idx+1:03}']
        eps = p[f'midplane_dust_to_gas_{idx+1:03}']
        l0 = l0 + 1 / (1 + St ** 2) * eps
        l1 = l1 + St / (1 + St ** 2) * eps
    p['lambda_0'] = l0
    p['lambda_1'] = l1

    v_P = p['velocity_pressure']
    v_visc = p['velocity_visc']
    l0 = p['lambda_0']
    l1 = p['lambda_1']

    # velocity_radial_gas is (11) in Dipierro+2018
    p['gas_velocity_radial'] = (-l1 * v_P + (1 + l0) * v_visc) / (
        (1 + l0) ** 2 + l1 ** 2
    )

    # velocity_azimuthal_gas is (12) in Dipierro+2018
    p['gas_velocity_azimuthal'] = (
        1 / 2 * (v_P * (1 + l0) + v_visc * l1) / ((1 + l0) ** 2 + l1 ** 2)
    )

    # velocity_radial_dust_i is (13) in Dipierro+2018
    # velocity_azimuthal_dust_i is (14) in Dipierro+2018
    for idx in range(num_dust):
        St = p[f'_midplane_stokes_number_{idx+1:03}']
        eps = p[f'midplane_dust_to_gas_{idx+1:03}']
        numerator_R = v_P * ((1 + l0) * St - l1) + v_visc * (1 + l0 + St * l1)
        numerator_phi = 0.5 * v_P * (1 + l0 + St * l1) - v_visc * ((1 + l0) * St - l1)
        denominator = ((1 + l0) ** 2 + l1 ** 2) * (1 + St ** 2)
        p[f'dust_velocity_radial_{idx+1:03}'] = numerator_R / denominator
        p[f'dust_velocity_azimuthal_{idx+1:03}'] = numerator_phi / denominator

    # Divide by |v_P| for comparison with Figure B1 in Dipierro+2018
    # "Analytical" solution
    v_R = p['gas_velocity_radial']
    p['gas_velocity_radial_analytical'] = v_R / np.abs(v_P)
    for idx in range(num_dust):
        v_R = p[f'dust_velocity_radial_{idx+1:03}']
        p[f'dust_velocity_radial_analytical_{idx+1:03}'] = v_R / np.abs(v_P)

    # "Numerical" solution
    v_R = p['velocity_radial_cylindrical']
    v_R_std = p['velocity_radial_cylindrical_std']
    p['velocity_radial_numerical'] = v_R / np.abs(v_P)
    p['velocity_radial_numerical_std'] = v_R_std / np.abs(v_P)
    for prof in profs['dust']:
        v_R = prof['velocity_radial_cylindrical']
        v_R_std = prof['velocity_radial_cylindrical_std']
        prof['velocity_radial_numerical'] = v_R / np.abs(v_P)
        prof['velocity_radial_numerical_std'] = v_R_std / np.abs(v_P)

    return profs


def plot_profiles(snap, profs, dust_species_to_plot, debug=False):
    """Plot radial drift velocity.

    Compares the numerical and analytical solutions.

    Parameters
    ----------
    snap
        The Snap object from which the profiles are generated.
    profs
        A dictionary of lists of profiles. See calculate_profiles.
    dust_species_to_plot
        The indices of the dust species to plot.
    debug
        A debug flag.

    Returns
    -------
    AxesSubplot
        A matplotlib AxesSubplot object.
    """
    print('Plotting profiles...')

    units = {
        'position': 'au',
        'gas_velocity_radial_analytical': 'dimensionless',
        'dust_velocity_radial_analytical': 'dimensionless',
        'velocity_radial_numerical': 'dimensionless',
    }
    p = profs['gas'][0]

    if debug:
        num_dust = snap.num_dust_species
        ax = p.plot(x='radius', y=['velocity_pressure', 'velocity_visc'], units=units)
        y = ['gas_velocity_radial']
        y += [f'dust_velocity_radial_{idx+1:03}' for idx in range(num_dust)]
        ax = p.plot(x='radius', y=y, units=units)
        ax.legend().remove()

    fig, ax = plt.subplots()

    # Plot "analytical" radial drift velocity / velocity pressure component
    p.plot(
        x='radius',
        y='gas_velocity_radial_analytical',
        units=units,
        color='black',
        label='',
        ax=ax,
    )
    y = [f'dust_velocity_radial_analytical_{idx+1:03}' for idx in dust_species_to_plot]
    p.plot(x='radius', y=y, units=units, label='', ax=ax)
    colors = [line.get_color() for line in ax.lines[1:]]

    # Plot "numerical" radial drift velocity / velocity pressure component
    p.plot(
        x='radius',
        y='velocity_radial_numerical',
        units=units,
        color='black',
        linestyle='',
        marker='o',
        markersize=4,
        fillstyle='none',
        label='gas',
        std='shading',
        ax=ax,
    )
    profs_to_plot = [
        prof for idx, prof in enumerate(profs['dust']) if idx in dust_species_to_plot
    ]
    for species, prof, color in zip(dust_species_to_plot, profs_to_plot, colors):
        label = f'{snap.properties["grain_size"][species].to("cm"):.1f~P}'
        prof.plot(
            x='radius',
            y='velocity_radial_numerical',
            units=units,
            color=color,
            linestyle='',
            marker='o',
            markersize=4,
            fillstyle='none',
            label=label,
            std='shading',
            ax=ax,
        )

    ax.set_ylabel(r'$v_R / |v_P|$')
    ax.grid()

    textstr = f't = {snap.properties["time"].to("years").m:.0f} years'
    bbox = dict(boxstyle='round', facecolor='white', edgecolor='grey', alpha=0.8)
    ax.text(
        0.97,
        0.97,
        textstr,
        transform=ax.transAxes,
        horizontalalignment='right',
        verticalalignment='top',
        bbox=bbox,
    )
    ax.legend(framealpha=0.8, edgecolor='grey')

    return ax


if __name__ == '__main__':
    snap, profs, ax = main(PARAMETERS)
