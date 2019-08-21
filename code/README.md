Code
====

This folder contains code for the three main test problems:

+ [DUSTYBOX](##DUSTYBOX)
+ [DUSTYWAVE](##DUSTYWAVE)
+ [DUSTYSHOCK](##DUSTYSHOCK)

DUSTYBOX
--------

Here is a brief description of the contents of the `DUSTYBOX` directory:

- `setup_dustybox.f90` contains the setup routine for `phantomsetup`
- `dustybox.patch` is a git patch containing the same information as
  `setup_dustybox.f90` in patch form
- `dustybox-Epstein-Stokes.in` and `dustybox-Kdrag.in` are the Phantom in files
  for Epstein/Stokes drag, and constant K drag
- `dustybox_analysis.py` and `dustybox_run.py` are Python scripts to run and
  analyse the DUSTYBOX test

### Run the test

To run a DUSTYBOX test, modify `dustybox_run.py` with the required parameters, then run with 

```
python dustybox_run.py
```

The script points to the run directory, so it *does not* have to be run from the actual Phantom run directory.

### Analyse results

To run the analysis, i.e. produce figures

```
python dustybox_analysis.py --directory /path/to/dustybox/run
```

DUSTYWAVE
---------

DUSTYSHOCK
----------
