Dustyshock
==========

Compile Phantom for dustyshock, and set up calculation. Here is an example
(assuming you are in the scripts directory).

```bash
python dustyshock_compile_phantom.py 'N=1_alpha=0' \
--run_root_dir ~/runs/multigrain/dustyshock \
--system gfortran \
--equation_of_state isothermal \
--fortran_compiler gfortran-9
```
