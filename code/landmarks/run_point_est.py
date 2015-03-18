#!/usr/bin/python
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
Perform a matching of point sets using mpp. Points can either be randomly
generated or pre specified.
"""

import __builtin__
__builtin__.__debug = True

import est as est
import matching.pointsim as ptsim
import numpy as np
import logging

#import pdb
#pdb.set_trace()

DIM = 2
SIGMA = 1.
WEIGHTS = [.1, .9]
WEIGHTS = WEIGHTS/np.sum(WEIGHTS)
maxIter = 100

logging.basicConfig(level=logging.DEBUG,format="[%(filename)s:%(lineno)s - %(funcName)6s() ] %(message)s")

random = False

# points
if random:
    Nsamples = 3
    N = 2
    rank = 0 # N*DIM
    qs = SIGMA*np.random.randn(Nsamples,N,DIM)
else:
    qs = np.array([\
            [[-.5 , 0.0], [.5,0.0]],\
            [[-.5 , 1.0], [.5,1.0] ]\
            ])
    N = qs.shape[1]
    Nsamples = qs.shape[0]
    rank = 1 # N*DIM

(mu,SigmaSQRT,Logyis) = est.est(qs,SIGMA,WEIGHTS,rank,visualize=True,visualizeIterations=False,maxIter=maxIter)

#print res

if True: # res.success:
    match.genStateData(fstate,sim)
