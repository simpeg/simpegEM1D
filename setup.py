#!/usr/bin/env python
from __future__ import print_function
"""simpegEM1D

simpegEM1D is the package for simulation and inversion of
electromagnetic data using 1D layered-earth solution.
"""

from distutils.core import setup
from setuptools import find_packages


CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
    'Natural Language :: English',
]

with open('README.md') as f:
    LONG_DESCRIPTION = ''.join(f.readlines())

setup(
    name='simpegEM1D',
    version='0.0.15',
    packages=find_packages(),
    install_requires=[
        'SimPEG>=0.4.1',
        'empymod>=1.6.2',
        'multiprocess'
    ],
    author='Seogi Kang',
    author_email='skang@eoas.ubc.ca',
    description='simpegEM1D',
    long_description=LONG_DESCRIPTION,
    keywords='geophysics, electromagnetics',
    url='https://github.com/simpeg/simpegEM1D',
    download_url='https://github.com/simpeg/simpegEM1D',
    classifiers=CLASSIFIERS,
    platforms=['Windows', 'Linux', 'Solaris', 'Mac OS-X', 'Unix'],
    license='MIT License',
    use_2to3=False,
)
