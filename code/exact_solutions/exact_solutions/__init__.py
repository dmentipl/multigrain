"""
Exact solutions for astrophysical problems.

This library contains exact solutions to astrophysical problems for
testing hydrodynamical codes.
"""

from . import dustybox
from . import dustywave

__version__ = '0.0.1'

__all__ = ['dustybox', 'dustywave']
