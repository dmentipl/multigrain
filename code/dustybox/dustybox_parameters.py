from dataclasses import dataclass

from numpy import ndarray
import pint


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


def set_parameters(parameter_dict):
    # TODO: docstring

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


if __name__ == '__main__':

    parameter_dict = {
        'prefix': 'dustybox',
        'length_unit': 'cm',
        'mass_unit': 'g',
        'time_unit': 's',
        'sound_speed': (1.0, 'cm / s'),
        'box_boundary': ([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5], 'cm'),
        'number_of_particles_gas': 50_000,
        'number_of_particles_dust': 10_000,
        'density_gas': (1.0e-13, 'g / cm^3'),
        'dust_method': 'largegrains',
        'drag_method': 'Epstein/Stokes',
        'grain_size': ([0.1, 0.316, 1.0, 3.16, 10.0], 'cm'),
        'grain_density': (0.5e-14, 'g / cm^3'),
        'velocity_delta': (1.0, 'cm / s'),
        'maximum_time': (0.1, 's'),
        'dust_to_gas_ratio': (0.02, 0.02, 0.02, 0.02, 0.02),
        'number_of_dumps': 100,
    }

    parameters = set_parameters(parameter_dict)
