#!/usr/bin/env python
"""Exploring non-linear inversions: a 1D magnetotelluric example

tle-magnetotelluric_inversion is a collection of functions for
exploring a 1D magnetotelluric example of non-linear inversions.

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

with open("README.rst") as f:
    LONG_DESCRIPTION = ''.join(f.readlines())

setup(
    name="tle-magnetotelluric_inversion",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        SimPEG
    ],
    author="Seogi Kang",
    author_email="skang@eoas.ubc.ca",
    description="Exploring non-linear inversions: a 1D magnetotelluric example",
    long_description=LONG_DESCRIPTION,
    license="MIT",
    keywords="electromagnetics geophysics",
    url="http://simpeg.xyz/",
    download_url="https://github.com/simpeg/tle-magnetotelluric_inversion",
    classifiers=CLASSIFIERS,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    use_2to3=False
)
