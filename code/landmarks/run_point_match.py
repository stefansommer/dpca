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

import mpp
import match as match
import matching.pointsim as ptsim
import numpy as np
import logging
from scipy import linalg

#import pdb
#pdb.set_trace()

DIM = mpp.DIM = 2
N = mpp.N = 2
SIGMA = mpp.SIGMA = 1.
WEIGHTS = [.0, 1.]
WEIGHTS = WEIGHTS/np.sum(WEIGHTS)
maxIter = 100
mpp.rank = N*DIM
mpp.init()

logging.basicConfig(level=logging.DEBUG,format="[%(filename)s:%(lineno)s - %(funcName)6s() ] %(message)s")

random = False

# moving points
if random:
    qm = SIGMA*np.random.randn(N,DIM)
else:
    #qm = 10.*np.array([[-1.0 , 0.0]])
    qm = 1.*np.array([[-.5 , 0.0],[.5,0.0]])
    #qm = 1.*np.array([[-1.0 , 0.0],[0.0,0.0],[1.0,0.0]])

# fixed points
if random:
    qf = SIGMA*np.random.randn(N,DIM)
else:
    #qf = 10.*np.array([[-1.0 , 1.0]])
    qf = 1.*np.array([[-1.5 , 1.0],[1.5,1.0]])
    #qf = 1.*np.array([[-1.0 , 1.0],[0.0,1.0],[1.0,1.0]])

moving=(qm, )
fixed=(qf, )

# specify frame/rank d map
x = qm
(gsharp,_,_,g,_) = mpp.gs(x)
#Xa = np.outer(np.ones(N),np.eye(DIM)).reshape([N,DIM,DIM])

# length of frame vectors (stds) in gsharp coordinates and covariances
stds = np.array([.5,2]) # more freedom in x direction  (for paper: (1,1), (.5,2), (2,.5))
covar = np.einsum('i,j->ji',stds**2,np.ones(N)).flatten()

mpp.lambdag = lambdag = np.min(covar) # metric part
covar = covar-lambdag # extract subtract part that metric will carry
inds = covar>0.
if np.any(inds):
    orthFrame = linalg.sqrtm(gsharp)
    Xa = np.einsum('ij,j->ij',orthFrame[:,inds],np.sqrt(covar[inds]))
    mpp.rank = rank = Xa.shape[1]
    assert(np.linalg.matrix_rank(Xa) == mpp.rank)
else:
    Xa = np.empty([N*DIM,0])
    mpp.rank = rank = 0
delta1 = np.eye(mpp.rank)
W = np.einsum('ab,ka,lb->kl',delta1,Xa,Xa)+lambdag*gsharp

assert(mpp.isFullRank() == (mpp.rank == N*DIM))
print "rank: %d, lambdag: %g, \nXa: \n%s, \nW: \n%s " % (mpp.rank,mpp.lambdag,Xa,W)



sim = ptsim.get(fixed, moving=moving, visualize=True)
sim['SIGMA'] = SIGMA

(fstate,res) = match.match(sim,Xa,SIGMA,WEIGHTS,lambdag,initial=moving,visualize=True,visualizeIterations=False,maxIter=maxIter)

#print res

if True: # res.success:
    match.genStateData(fstate,sim)
