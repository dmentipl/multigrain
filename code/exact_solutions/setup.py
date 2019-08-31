import io
import pathlib
import re

from setuptools import setup

version = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open('exact_solutions/__init__.py', encoding='utf_8_sig').read(),
).group(1)

long_description = (pathlib.Path(__file__).parent / 'README.md').read_text()

setup(
    name='exact_solutions',
    version=version,
    author='Daniel Mentiplay',
    packages=['exact_solutions'],
    license='MIT',
    description='Exact solutions for astrophysical problems',
    long_description=long_description,
    long_description_content_type='text/markdown',
)

