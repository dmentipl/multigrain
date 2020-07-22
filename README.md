A smoothed particle hydrodynamics algorithm for multigrain dust with separate sets of particles
---

by Daniel Mentiplay, Daniel Price, and Christophe Pinte

We plan to submit the manuscript to *Monthly Notices of the Royal Astronomical Society*.
> (FOR FUTURE REFERENCE) CITATION GOES HERE

**This repository contains the data and code used to produce all results and figures shown in the paper.** An archived version of this repository will be available at Figshare.
> (FOR FUTURE REFERENCE) Figshare LINK GOES HERE

We aim to describe multigrain dust methods in smoothed particle hydrodynamics using separate sets of particles for each dust species. We test these methods against standard dust-gas hydrodynamics tests, including a dusty-box, a dusty-wave, and a dusty-shock.

Abstract
--------

> ABSTRACT GOES HERE

Software
--------

We implemented the multigrain dust methods in [Phantom](https://github.com/danieljprice/phantom). In fact, this method for multigrain dust has been in the Phantom master branch since [`64dbd2b1`](https://github.com/danieljprice/phantom/commit/64dbd2b124ca74051eed920d6cad0a2e83157478), September 18, 2018. For this manuscript, the focus is writing test problems.

### Setting up the environment

All the code added to Phantom is in the code directory, as are Python scripts to run the tests, and to perform analysis on them. The `environment.yml` file with this repository contains the Python required packages. To use this file to create a Conda environment for this work, and then use it, do

```bash
conda env create --file environment.yml
conda activate multigrain
jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib
jupyter nbextension enable --py widgetsnbextension
```

Results
-------

We use Phantom version `666da9e892cb3f2d9f89e132504e185fe2f22f31` with some patches in the [`code/patches`]((https://github.com/dmentipl/multigrain/tree/master/code/patches)) directory. As long as the conda environment (described above) is set up, the scripts to set up and run the tests and analyse the results should do everything. I.e. you don't need to manually clone Phantom and apply patches. Nor do you have to run `phantom` and `phantomsetup` manually. The scripts *should* do this.

### Running the Phantom multigrain tests

There are Python scripts to setup and run the tests located in [`code/scripts`](https://github.com/dmentipl/multigrain/tree/master/code/scripts).

#### Dusty-box

The dusty-box tests are few and quick and can be run on a local machine. The data generated is about 1.1 GB in size.

- `dustybox_setup_and_run_time_evolution.py` runs 6 tests: 3 with total dust-to-gas ratio of 0.01, and 3 with total dust-to-gas ratio of 0.5. In each triple there is 1 dust species, 2 dust species, and 5 dust species.
- `dustybox_setup_and_run_stability.py` to check the stability criterion.

#### Dusty-wave

The dusty-wave tests are few and quick and can be run on a local machine. The data generated is about 250 MB in size.

- `dustywave_setup_and_run.py` runs 2 tests: one with 1 dust species and one with 4 dust species.

#### Dusty-shock

The dusty-shock tests are many and slow and should be run on a cluster.

- `dustyshock_setup_and_schedule.py` runs a slew of tests. This relies on `dustyshock.toml.j2` which is a [phantom-build](https://github.com/dmentipl/phantom-build) config file.

### Performing analysis on the tests

There are notebooks in [`code/notebooks`](https://github.com/dmentipl/multigrain/tree/master/code/notebooks) to perform analysis on the Phantom output.

### Manuscript figures

Python scripts for generating the manuscript figures (after running the tests above) are available in [`code/scripts`](https://github.com/dmentipl/multigrain/tree/master/code/scripts).

- `dustybox_figures.py`
- `dustywave_figures.py`
- `dustyshock_figures.py`

Manuscript
----------

Uses the MNRAS template from Overleaf.

Make the manuscript with

```bash
make manuscript
```
