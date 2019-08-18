#!/usr/bin/env python
from __future__ import print_function
"""simpegEM1D

simpegEM1D is the package for simulation and inversion of
electromagnetic data using 1D layered-earth solution.
"""
import setuptools
from numpy.distutils.core import setup, Extension
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

fExt = [Extension(name='simpegEM1D.m_rTE_Fortran', # Name of the package to import
                  sources=['simpegEM1D/Fortran/m_rTE_Fortran.f90'],
                #   extra_f90_compile_args=['-ffree-line-length-none', #This if for intel-fortran
                #                       '-O3',
                #                       '-finline-functions',
                #                       '-funroll-all-loops',
                #                       '-DNDEBUG',
                #                       '-g0'],
                  extra_link_args=['-ffree-line-length-none',
                                      '-O3',
                                      '-finline-functions',
                                      '-funroll-all-loops',
                                      '-g0'],
                  )
        ]


setup(
    name='simpegEM1D',
    version='0.0.18a',
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
    ext_modules=fExt
)
