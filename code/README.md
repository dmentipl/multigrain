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

```bash
cd exact_solutions && pip install --no-deps -e .
```

This may be open-sourced in the future.

Scripts
-------

To set up and run all the Phantom tests

```bash
make run
```

To analyse the results of the tests, i.e. produce figures, etc.

```bash
make analysis
```
