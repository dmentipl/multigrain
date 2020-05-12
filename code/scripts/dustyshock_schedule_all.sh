#!/usr/bin/env bash

python -c 'import pathlib' || (echo 'Must be Python 3' && exit)

python dustyshock_compile_phantom.py 'N=1_nx=64' \
  --run_root_dir /fred/oz015/dmentipl/runs/multigrain/dustyshock \
  --system ifort \
  --equation_of_state isothermal \
  --schedule_job

python dustyshock_compile_phantom.py 'N=1_nx=128' \
  --run_root_dir /fred/oz015/dmentipl/runs/multigrain/dustyshock \
  --system ifort \
  --equation_of_state isothermal \
  --schedule_job

python dustyshock_compile_phantom.py 'N=1_nx=256' \
  --run_root_dir /fred/oz015/dmentipl/runs/multigrain/dustyshock \
  --system ifort \
  --equation_of_state isothermal \
  --schedule_job

python dustyshock_compile_phantom.py 'N=3_nx=64' \
  --run_root_dir /fred/oz015/dmentipl/runs/multigrain/dustyshock \
  --system ifort \
  --equation_of_state isothermal \
  --schedule_job

python dustyshock_compile_phantom.py 'N=3_nx=128' \
  --run_root_dir /fred/oz015/dmentipl/runs/multigrain/dustyshock \
  --system ifort \
  --equation_of_state isothermal \
  --schedule_job

python dustyshock_compile_phantom.py 'N=3_nx=256' \
  --run_root_dir /fred/oz015/dmentipl/runs/multigrain/dustyshock \
  --system ifort \
  --equation_of_state isothermal \
  --schedule_job

python dustyshock_compile_phantom.py 'N=1_alpha=0' \
  --run_root_dir /fred/oz015/dmentipl/runs/multigrain/dustyshock \
  --system ifort \
  --equation_of_state isothermal \
  --schedule_job

python dustyshock_compile_phantom.py 'N=3_alpha=0' \
  --run_root_dir /fred/oz015/dmentipl/runs/multigrain/dustyshock \
  --system ifort \
  --equation_of_state isothermal \
  --schedule_job

python dustyshock_compile_phantom.py 'N=1_recon_off' \
  --run_root_dir /fred/oz015/dmentipl/runs/multigrain/dustyshock \
  --system ifort \
  --equation_of_state isothermal \
  --schedule_job

python dustyshock_compile_phantom.py 'N=3_recon_off' \
  --run_root_dir /fred/oz015/dmentipl/runs/multigrain/dustyshock \
  --system ifort \
  --equation_of_state isothermal \
  --schedule_job
