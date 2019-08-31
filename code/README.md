Code
====

This folder contains code for the three main test problems:

+ [DUSTYBOX](DUSTYBOX)
+ [DUSTYWAVE](DUSTYWAVE)
+ [DUSTYSHOCK](DUSTYSHOCK)

To run all the Phantom tests

```
make run-tests
```

Exact solutions
---------------

This directory contains a Python library `exact_solutions` with exact solutions to the test problems. To build the package type

```
cd exact_solutions && pip install --no-deps -e .
```

DUSTYBOX
--------

Here is a brief description of the contents of the `DUSTYBOX` directory:

- `setup_dustybox.f90` contains the setup routine for `phantomsetup`
- `dustybox.patch` is a git patch containing the same information as
  `setup_dustybox.f90` in patch form
- `templates` contains Phantom in file templates
- `dustybox_analysis.py` and `dustybox_run.py` are Python scripts to run and
  analyse the DUSTYBOX test

### Run the tests

To run the DUSTYBOX tests

```
make dustybox-run
```

### Analyse results

To run the analysis, i.e. produce figures

```
python dustybox_analysis.py --directory /path/to/dustybox/run
```

DUSTYWAVE
---------

DUSTYSHOCK
----------
