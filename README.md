Large grain multigrain: smoothed particle hydrodynamics for large dust grains
=============================================================================

by Daniel Mentiplay, Daniel Price, Guillaume Laibe, and Christophe Pinte

We plan to submit the manuscript to *Monthly Notices of the Royal Astronomical Society*.
> (FOR FUTURE REFERENCE) CITATION GOES HERE

**This repository contains the data and code used to produce all results and figures shown in the paper.** An archived version of this repository will be available at Figshare.
> (FOR FUTURE REFERENCE) FIGSHARE LINK GOES HERE

We aim to describe large grain multigrain dust methods in smoothed particle hydrodynamics. We test these methods against standard dust-gas hydrodynamics tests, including a dusty-box, a dusty-wave, and a dusty-shock.

Abstract
--------

> ABSTRACT GOES HERE

Software
--------

We implemented the multigrain dust methods in [Phantom](https://bitbucket.org/danielprice/phantom/). In fact, the method for large grain multigrain has been in the Phantom master branch since `64dbd2b1`, September 18, 2018.  For this manuscript, the focus is writing test problems, working from Phantom git SHA version `6666c55f`.

See the `README.md` in the code directory for more details on the software used in this project.

### Setting up the environment

All the code added to Phantom is in the code directory, as are Python scripts to run the tests, and to perform analysis on them. The `environment.yml` file with this repository contains the Python required packages. To use this file to create a Conda environment for this work, and then use it, do

```bash
conda env create -f environment.yml
conda activate multigrain
```

### Exact solutions

You need to locally pip install a Python library for exact solutions contained within this repository. From this directory, do

```bash
cd code/exact_solutions && pip install --no-deps -e .
```

Results
-------

### Running the Phantom multigrain tests

To run the Phantom multigrain dust tests:

```bash
make run
```

This will get the required version of Phantom, apply any patches, compile Phantom for each problem, create directories with different parameters, setup the Phantom calculations, then run them.

### Performing analysis on the tests

To perform analysis on the Phantom output:

```bash
make analysis
```

This will read the Phantom output, post-process it, and produce figures, and CSV files as required for the particular problem.
