Code for producing results and figures
======================================

This folder contains code for the three main test problems:

* DUSTYBOX
* DUSTYWAVE
* DUSTYSHOCK

as well as a Python library `exact_solutions`.

Exact solutions library
-----------------------

This directory contains a Python library `exact_solutions` with exact solutions to the test problems. To build the package type

```
cd exact_solutions && pip install --no-deps -e .
```

This may be open-sourced in the future.

Scripts
-------

To run all the Phantom tests

```
make run
```

To analyse the results of the tests, i.e. produce figures, etc.

```
make analysis
```

Test problems
-------------

### DUSTYBOX

Here is a brief description of the contents of the `DUSTYBOX` directory:

- `setup_dustybox.f90` contains the setup routine for `phantomsetup`
- `dustybox.patch` is a git patch containing the same information as
  `setup_dustybox.f90` in patch form
- `templates` contains Phantom in file templates
- `dustybox_analysis.py` and `dustybox_run.py` are Python scripts to run and
  analyse the DUSTYBOX test

### DUSTYWAVE

### DUSTYSHOCK
