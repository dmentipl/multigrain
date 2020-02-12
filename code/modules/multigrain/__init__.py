"""Test problems for multigrain paper."""

from . import dustybox, dustywave, dustyshock
from .run import setup_multiple_calculations, run_multiple_calculations

__all__ = [
    'dustybox',
    'dustywave',
    'dustyshock',
    'setup_multiple_calculations',
    'run_multiple_calculations',
]
