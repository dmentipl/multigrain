#!/usr/bin/env bash

python -c 'import pathlib' || (echo 'Must be Python 3' && exit)

ROOT_DIR=/fred/oz015/dmentipl/runs/multigrain/dustyshock
SYSTEM=ifort
EQUATION_OF_STATE=isothermal

python dustyshock_compile_phantom.py \
  --run_name 'N=1_nx=64' \
  --run_name 'N=1_nx=128' \
  --run_name 'N=1_nx=256' \
  --run_name 'N=3_nx=64' \
  --run_name 'N=3_nx=128' \
  --run_name 'N=3_nx=256' \
  --run_name 'N=1_alpha=0' \
  --run_name 'N=3_alpha=0' \
  --run_name 'N=1_recon_off' \
  --run_name 'N=3_recon_off' \
  --root_dir $ROOT_DIR \
  --system $SYSTEM \
  --equation_of_state $EQUATION_OF_STATE \
  --schedule_job
