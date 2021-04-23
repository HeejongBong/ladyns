#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pkg_resources import parse_version
import os, sys

numpy_min_version = '1.8'

def get_numpy_status():

    """
    Returns a dictionary containing a boolean specifying whether NumPy
    is up-to-date, along with the version string (empty string if
    not installed).
    """

    numpy_status = {}
    try:
        import numpy
        numpy_version = numpy.__version__
        numpy_status['up_to_date'] = parse_version(
            numpy_version) >= parse_version(numpy_min_version)
        numpy_status['version'] = numpy_version
    except ImportError:
        numpy_status['up_to_date'] = False
        numpy_status['version'] = ""
    return numpy_status

def setup_ladyns():
    numpy_status = get_numpy_status()
    numpy_req_str = "LaDynS requires NumPy >= {0}.\n".format(numpy_min_version)      

    if numpy_status['up_to_date'] is False:
        if numpy_status['version']:
            raise ImportError("Your installation of NumPy"
                              "{0} is out-of-date.\n{1}"
                              .format(numpy_status['version'], numpy_req_str))
        else:
            raise ImportError("NumPy is not installed.\n{0}"
                              .format(numpy_req_str))   

    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.core import setup, Extension
    from numpy.distutils.system_info import get_info

    lapack_info = get_info('lapack_opt',1)
    sources = ['ladyns/optimize/core/lasso.f90',
               'ladyns/optimize/core/glasso.f90', 
               'ladyns/optimize/core/glasso_p.f90']

    if lapack_info:  
        ext = Extension(name='ladyns.optimize.core', sources=sources,
                        **lapack_info)

    if not lapack_info:
        raise('No BLAS or Lapack libraries found')

    setup(
        name='ladyns',
        version="1.0.0",
        description="LaDynS: Latent Dynamic Analysis via Sparse Banded Graphs",
        author="Heejong Bong",
        author_email="hbong@andrew.cmu.edu",
        url="http://github.com/HeejongBong/ladyns",
        license="MIT License",
        classifiers=[
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Fortran',
            'Topic :: Scientific/Engineering',
            ],
        package_dir = {
            'ladyns': 'ladyns',
            'ladyns.optimize': 'ladyns/optimize'},
        packages = ['ladyns', 'ladyns.optimize'],
        install_requires = ['numpy', 'matplotlib', 'scipy'],
        ext_modules = [ext])

if __name__ == '__main__':
    setup_ladyns()