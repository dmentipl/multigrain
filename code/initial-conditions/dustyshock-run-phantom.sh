#!/usr/bin/env bash

NAME=$1
echo "NAME='$NAME'"
[ -f ~/repos/multigrain/code/initial-conditions/dustyshock-"$NAME".setup ] || exit

# Set up calculation
mkdir ~/runs/multigrain/dustyshock/"$NAME"
cd ~/runs/multigrain/dustyshock/"$NAME" || exit
cp -f ~/repos/phantom/bin/phantom{,setup,_version} .
cp -f ~/repos/multigrain/code/initial-conditions/dustyshock-"$NAME".setup dustyshock.setup
cp -f ~/repos/multigrain/code/initial-conditions/dustyshock-"$NAME".in dustyshock.in
./phantomsetup dustyshock | tee dustyshock00.log 2>&1

# Run calculation
./phantom dustyshock.in | tee dustyshock01.log 2>&1
