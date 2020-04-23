Compiling Phantom for dustyshock
================================

Checkout, patch, compile Phantom.

```bash
cd ~/repos/phantom
git checkout master
git checkout -- '*'
git pull
git checkout d9a5507f3fd97b5ed5acf4547f82449476b29091
git apply ~/repos/multigrain/code/patches/phantom-d9a5507f-idustbound.patch
git apply ~/repos/multigrain/code/patches/phantom-d9a5507f-multigrain_setup_shock.patch
make \
    SETUP=dustyshock \
    SYSTEM=gfortran \
    FC=gfortran-9 \
    HDF5=yes \
    HDF5ROOT=$HDF5_DIR \
    phantom
make \
    SETUP=dustyshock \
    SYSTEM=gfortran \
    FC=gfortran-9 \
    HDF5=yes \
    HDF5ROOT=$HDF5_DIR \
    setup
```

First jump into tmux.

```bash
tmux
```

Set up and run calculations.

```bash
mkdir ~/runs/multigrain/dustyshock/N=1
cd ~/runs/multigrain/dustyshock/N=1
cp -f ~/repos/phantom/bin/phantom{,setup,_version} .
cp -f ~/repos/multigrain/code/initial-conditions/dustyshock-N=1.setup dustyshock.setup
cp -f ~/repos/multigrain/code/initial-conditions/dustyshock-N=1.in dustyshock.in
./phantomsetup dustyshock | tee dustyshock00.log 2>&1
./phantom dustyshock.in | tee dustyshock01.log 2>&1
```

```bash
mkdir ~/runs/multigrain/dustyshock/N=3
cd ~/runs/multigrain/dustyshock/N=3
cp -f ~/repos/phantom/bin/phantom{,setup,_version} .
cp -f ~/repos/multigrain/code/initial-conditions/dustyshock-N=3.setup dustyshock.setup
cp -f ~/repos/multigrain/code/initial-conditions/dustyshock-N=3.in dustyshock.in
./phantomsetup dustyshock | tee dustyshock00.log 2>&1
./phantom dustyshock.in | tee dustyshock01.log 2>&1
```
