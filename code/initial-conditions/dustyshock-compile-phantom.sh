#!/usr/bin/env bash

# Checkout
cd ~/repos/phantom || exit
git checkout -- '*'
git checkout master
git pull
git checkout d9a5507f3fd97b5ed5acf4547f82449476b29091

# Patch
git apply ~/repos/multigrain/code/patches/phantom-d9a5507f-idustbound.patch
git apply ~/repos/multigrain/code/patches/phantom-d9a5507f-multigrain_setup_shock.patch

# Compile
make \
    SETUP=dustyshock \
    SYSTEM=gfortran \
    FC=gfortran-9 \
    ISOTHERMAL=yes \
    HDF5=yes \
    HDF5ROOT=$HDF5_DIR \
    phantom
make \
    SETUP=dustyshock \
    SYSTEM=gfortran \
    FC=gfortran-9 \
    ISOTHERMAL=yes \
    HDF5=yes \
    HDF5ROOT=$HDF5_DIR \
    setup
