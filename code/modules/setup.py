"""Setup.py for multigrain."""

from setuptools import setup

setup(
    name='multigrain',
    packages=['multigrain'],
    install_requires=[
        'bokeh',
        'matplotlib',
        'numba',
        'numpy',
        'pandas',
        'phantomsetup',
        'pint',
        'plonk',
        'scipy',
    ],
    python_requires='>=3.7',
)
