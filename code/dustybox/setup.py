"""
Setup dustybox calculations.
"""

import pathlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pint
from numpy import ndarray

import phantomsetup


@dataclass
class Parameters:
    """
    Dusty box parameters.

        prefix: str
        length_unit: float
        mass_unit: float
        time_unit: float
        sound_speed: float
        box_boundary: tuple
        number_of_particles_gas: int
        number_of_particles_dust: int
        density_gas: float
        dust_to_gas_ratio: tuple
        dust_method: str
        drag_method: str
        velocity_delta: float
        maximum_time: float
        K_drag: float = 0.0
        grain_size: tuple = ()
        grain_density: float = 0.0
        number_of_dumps: int = 100

    All dimensional quantities are in terms of the base units:

    - length_unit
    - mass_unit
    - time_unit

    Options for dust_method are:
    
    - 'largegrains'
    - 'smallgrains'

    Options for drag_method are:
    
    - 'Epstein/Stokes'
    - 'K_const'

    If drag_method is set to 'Epstein/Stokes' you must set
    
    - grain_size
    - grain_density

    If drag_method is set to 'K_drag' you must set K_drag.

    Note that the length of dust_to_gas_ratio must match the length
    grain_size and grain_density.
    """

    prefix: str
    length_unit: float
    mass_unit: float
    time_unit: float
    sound_speed: float
    box_boundary: tuple
    number_of_particles_gas: int
    number_of_particles_dust: int
    density_gas: float
    dust_to_gas_ratio: tuple
    dust_method: str
    drag_method: str
    velocity_delta: float
    maximum_time: float
    K_drag: float = 0.0
    grain_size: tuple = ()
    grain_density: float = 0.0
    number_of_dumps: int = 100

    def validate_input(self):
        if self.dust_method not in ('largegrains', 'smallgrains'):
            raise ValueError(
                'dust_method unavailable: choose "largegrains" or "smallgrains"'
            )
        if self.drag_method not in ('Epstein/Stokes', 'K_const'):
            raise ValueError(
                'drag_method unavailable: choose "Epstein/Stokes" or "K_const"'
            )
        if self.drag_method == 'Epstein/Stokes':
            if len(self.grain_size) != len(self.dust_to_gas_ratio):
                raise ValueError('len(grain_size) must equal len(dust_to_gas_ratio)')
            if len(self.grain_size) < 1:
                raise ValueError('must set grain_size if using Epstein/Stokes drag')
            if not self.grain_density > 0.0:
                raise ValueError('must set grain_density if using Epstein/Stokes drag')
        if self.drag_method == 'K_const':
            if not self.K_drag > 0.0:
                raise ValueError('must set K_drag if using K_const drag')

    def __post_init__(self):
        self.validate_input()


def set_parameters(parameter_dict: Dict[str, Any]) -> Parameters:
    """
    Generate Parameters object from dictionary.

    Parameters
    ----------
    parameters_dict
        A dictionary with keys the same as attributes in Parameters. The
        values can be float, int, list, tuple, str. Tuples are special.
        They indicate a numerical value (or array_like of values), with
        the last element of the tuple a str with the units.

    Returns
    -------
    Parameters
        The Parameters object.

    Examples
    --------
    >>> _parameters_asdict = {
    ...    'prefix': 'dustybox',
    ...    'length_unit': 'cm',
    ...    'mass_unit': 'g',
    ...    'time_unit': 's',
    ...    'sound_speed': (1.0, 'cm / s'),
    ...    'box_boundary': ([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5], 'cm'),
    ... }
    >>> set_parameters(_parameters_asdict)
    """

    units = pint.UnitRegistry(system='cgs')

    length_unit = units(parameter_dict.pop('length_unit'))
    mass_unit = units(parameter_dict.pop('mass_unit'))
    time_unit = units(parameter_dict.pop('time_unit'))

    dict_ = {
        'length_unit': length_unit.to_base_units().magnitude,
        'mass_unit': mass_unit.to_base_units().magnitude,
        'time_unit': time_unit.to_base_units().magnitude,
    }
    for key, value in parameter_dict.items():
        if isinstance(value, tuple) and isinstance(value[-1], str):
            quantity = units.Quantity(*value)
            d = quantity.dimensionality
            new_units = (
                length_unit ** d['[length]']
                * mass_unit ** d['[mass]']
                * time_unit ** d['[time]']
            )
            magnitude = quantity.to(new_units).magnitude
            if isinstance(magnitude, ndarray):
                dict_[key] = tuple(magnitude)
            else:
                dict_[key] = magnitude
        else:
            if isinstance(value, (list, ndarray)):
                dict_[key] = tuple(value)
            else:
                dict_[key] = value

    return Parameters(**dict_)


def do_setup(
    run_root_directory: Path,
    parameters_dict: Dict[str, Parameters],
    phantom_dir: Path,
    hdf5root: Path,
) -> List[phantomsetup.Setup]:
    """
    Setup multiple calculations.

    Parameters
    ----------
    run_root_directory
        The path to the root directory for this series of runs.
    parameters_dict
        A dictionary of Parameters. The key is the "run label" which
        will be the sub-directory of the root directory. The value is
        the Parameters data object for the run.
    phantom_dir
        The path to the Phantom repository.
    hdf5root
        The path to the root directory containing the HDF5 library.

    Returns
    -------
    List[phantomsetup.Setup]
        A list of Setup objects.
    """

    print('\n' + 72 * '-')
    print('>>> Setting up calculations <<<')
    print(72 * '-' + '\n')

    if not run_root_directory.exists():
        run_root_directory.mkdir(parents=True)

    setups = list()
    for run_label, parameters in parameters_dict.items():
        print(f'Setting up {run_label}...')
        run_directory = run_root_directory / run_label
        run_directory.mkdir()
        setups.append(
            setup_dustybox(
                parameters=parameters,
                run_directory=run_directory,
                phantom_dir=phantom_dir,
                hdf5root=hdf5root,
            )
        )

    return setups


def setup_dustybox(
    parameters: Parameters,
    run_directory: Union[str, Path],
    phantom_dir: Path,
    hdf5root: Path,
) -> phantomsetup.Setup:
    """
    Setup a Phantom dustybox calculation.

    Parameters
    ----------
    parameters
        The parameters for this calculation.
    run_directory
        The path to the directory containing the run.
    phantom_dir
        The path to the Phantom repository.
    hdf5root
        The path to the root directory containing the HDF5 library.

    Returns
    -------
    phantomsetup.Setup
    """

    # Constants
    igas = phantomsetup.defaults.PARTICLE_TYPE['igas']
    idust = phantomsetup.defaults.PARTICLE_TYPE['idust']

    # Setup
    setup = phantomsetup.Setup()
    setup.prefix = parameters.prefix

    setup.set_compile_option('IND_TIMESTEPS', False)
    setup.set_output(
        tmax=parameters.maximum_time, ndumps=parameters.number_of_dumps, nfulldump=1
    )

    if isinstance(parameters.length_unit, str):
        length_unit = phantomsetup.units.unit_string_to_cgs(parameters.length_unit)
    else:
        length_unit = parameters.length_unit
    if isinstance(parameters.mass_unit, str):
        mass_unit = phantomsetup.units.unit_string_to_cgs(parameters.mass_unit)
    else:
        mass_unit = parameters.mass_unit
    if isinstance(parameters.time_unit, str):
        time_unit = phantomsetup.units.unit_string_to_cgs(parameters.time_unit)
    else:
        time_unit = parameters.time_unit
    setup.set_units(length=length_unit, mass=mass_unit, time=time_unit)

    setup.set_equation_of_state(ieos=1, polyk=parameters.sound_speed ** 2)

    number_of_dust_species = len(parameters.dust_to_gas_ratio)
    density_dust = [
        eps * parameters.density_gas for eps in parameters.dust_to_gas_ratio
    ]
    if parameters.drag_method == 'Epstein/Stokes':
        setup.set_dust(
            dust_method=parameters.dust_method,
            drag_method=parameters.drag_method,
            grain_size=parameters.grain_size,
            grain_density=parameters.grain_density,
        )
    elif parameters.drag_method == 'K_const':
        setup.set_dust(
            dust_method=parameters.dust_method,
            drag_method=parameters.drag_method,
            drag_constant=parameters.K_drag,
            number_of_dust_species=number_of_dust_species,
        )
    else:
        raise ValueError('Cannot set up dust')

    setup.set_boundary(parameters.box_boundary, periodic=True)

    def velocity_gas(xyz: ndarray) -> ndarray:
        """Gas has zero initial velocity."""
        vxyz = np.zeros_like(xyz)
        return vxyz

    box = phantomsetup.Box(*parameters.box_boundary)
    box.add_particles(
        particle_type=igas,
        number_of_particles=parameters.number_of_particles_gas,
        density=parameters.density_gas,
        velocity_distribution=velocity_gas,
    )
    setup.add_box(box)

    def velocity_dust(xyz: ndarray) -> ndarray:
        """Dust has uniform initial velocity."""
        vxyz = np.zeros_like(xyz)
        vxyz[:, 0] = parameters.velocity_delta
        return vxyz

    for idx in range(number_of_dust_species):
        box = phantomsetup.Box(*parameters.box_boundary)
        box.add_particles(
            particle_type=idust + idx,
            number_of_particles=parameters.number_of_particles_dust,
            density=density_dust[idx],
            velocity_distribution=velocity_dust,
        )
        setup.add_box(box)

    alpha = np.zeros(setup.total_number_of_particles, dtype=np.single)
    setup.add_array_to_particles('alpha', alpha)

    # Write to file
    setup.write_dump_file(directory=run_directory)
    setup.write_in_file(directory=run_directory)

    # Compile Phantom
    setup.compile_phantom(
        phantom_dir=phantom_dir, hdf5root=hdf5root, working_dir=run_directory
    )

    # Return setup
    return setup
