#!/usr/bin/env python
#
# This file is part of jetflows.
#
# Copyright (C) 2014, Henry O. Jacobs (hoj201@gmail.com), Stefan Sommer (sommer@di.ku.dk)
# https://github.com/nefan/jetflows.git
#
# jetflows is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# jetflows is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with jetflows.  If not, see <http://www.gnu.org/licenses/>.
#


"""
setup.py file for SWIG example
"""

from distutils.core import *
from distutils import sysconfig

import numpy


try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()    




gaussian_module = Extension('_gaussian',
                           sources=['gaussian.i'],
                           include_dirs = [numpy_include,'/home/stefan/projects/sla'],
                           swig_opts=['-c++'],  #enable c++ wrapping
                           extra_compile_args = ['-fopenmp','-fpic'],
                           extra_link_args = ['-lgomp']
                           )

setup (name = 'gaussian',
       version = '0.1',
       author      = "Stefan Sommer",
       description = """Evaluate Gaussian kernel for jet particle simulations""",
       ext_modules = [gaussian_module],
       py_modules = ["gaussian"],
       )
