Hybrid multigrain: A smoothed particle hydrodynamics algorithm for small and large dust grains
==============================================================================================

by Daniel Mentiplay, Daniel Price, Guillaume Laibe, and Christophe Pinte

The paper is to be submitted to *Monthly Notices of the Royal Astronomical Society*. The citation is
> (FOR FUTURE REFERENCE) CITATION GOES HERE

**This repository contains the data and code used to produce all results and figures shown in the paper.** An archived version of this repository will be available at Figshare
> (FOR FUTURE REFERENCE) FIGSHARE LINK GOES HERE

We aim to describe hybrid multigrain dust methods in smoothed particle hydrodynamics. We test these methods against standard dust-gas hydrodynamics tests, including DUSTYBOX, DUSTYWAVE, and a dusty shock.

Abstract
--------

> ABSTRACT GOES HERE

Software
--------

We implement the multigrain dust methods in [Phantom](https://bitbucket.org/danielprice/phantom/wiki/Home). In fact, the method for large grain multigrain has been in the Phantom master branch since `64dbd2b1`, September 18, 2018.  For this manuscript, the focus is writing test problems, working from Phantom git SHA version `6666c55f`.

All the code added to Phantom is in the code directory, as are Python scripts to run the tests, and to perform analysis on them. There is an `environment.yml` file with this repository that contains the Python required packages. To use this file to create a Conda environment for this work, and then use it, do

```
conda env create -f environment.yml
conda activate multigrain
```

Results
-------

### Running the Phantom multigrain tests

To run the Phantom multigrain dust tests:

```
make run-tests
```

This will compile Phantom for each problem, create directories with different parameters, setup the Phantom calculations, then run them.

### Performing analysis on the tests

To perform analysis on the Phantom output:

```
make analyse-tests
```

For the DUSTYBOX test, this computes the differential velocity of the dust for each species to compare with exact solutions.
