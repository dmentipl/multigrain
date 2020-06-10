"""Generate Phantom .in and .setup files for dustyshock."""

from pathlib import Path

from phantomconfig import read_config

DATA_PATH = Path(__file__).parent
GENERATED_PATH = Path('~/runs/multigrain/dustyshock/_initial_conditions').expanduser()

# .setup and .in template files
SETUPFILE_TEMPLATE = DATA_PATH / 'dustyshock.setup'
INFILE_TEMPLATE = {
    1: DATA_PATH / 'dustyshock-N_1.in',
    3: DATA_PATH / 'dustyshock-N_3.in',
}

# Parameters to loop over
NDUST = [1, 3]
NX = [32, 128]
SMOOTH_FAC = [2.0, 5.0]
HFACT = [1.0, 1.2, 1.5]

# Shock parameters per ndust
SHOCK_PARAMS = {
    1: {'densright': 8.0, 'prright': 8.0, 'vxright': 0.25},
    3: {'densright': 16.0, 'prright': 16.0, 'vxright': 0.125},
}


def main():
    """Run main script function."""
    for ndust in NDUST:
        for nx in NX:
            for smooth_fac in SMOOTH_FAC:
                for hfact in HFACT:
                    name = f'N_{ndust}-nx_{nx}-smooth_fac_{smooth_fac}-hfact_{hfact}'
                    new_path = GENERATED_PATH / name
                    if not new_path.exists():
                        new_path.mkdir(parents=True)
                    generate_setup_file(ndust, nx, smooth_fac, new_path)
                    generate_in_file(ndust, hfact, new_path)


def generate_setup_file(ndust, nx, smooth_fac, path):
    """Generate a .setup file."""
    setupfile = read_config(SETUPFILE_TEMPLATE)
    setupfile.change_value('ndust', ndust)
    setupfile.change_value('nx', nx)
    setupfile.change_value('smooth_fac', smooth_fac)
    setupfile.change_value('densright', SHOCK_PARAMS[ndust]['densright'])
    setupfile.change_value('prright', SHOCK_PARAMS[ndust]['prright'])
    setupfile.change_value('vxright', SHOCK_PARAMS[ndust]['vxright'])
    setupfile.write_phantom(path / 'dustyshock.setup')


def generate_in_file(ndust, hfact, path):
    """Generate a .in file."""
    infile = read_config(INFILE_TEMPLATE[ndust])
    infile.change_value('hfact', hfact)
    infile.write_phantom(path / 'dustyshock.in')


if __name__ == '__main__':
    main()
