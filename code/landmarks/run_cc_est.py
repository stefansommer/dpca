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
import match as match
import matching.pointsim as ptsim
import numpy as np
import logging

#import pdb
#pdb.set_trace()

DIM = 2
SIGMA = .1
WEIGHTS = [.1, .9]
WEIGHTS = WEIGHTS/np.sum(WEIGHTS)
maxIter = 10
rank = 1

logging.basicConfig(level=logging.DEBUG,format="[%(filename)s:%(lineno)s - %(funcName)6s() ] %(message)s")

# load data
from scipy import io
data=io.loadmat('../../data/dataM-corpora-callosa-ipmi-full.mat')
vi = data['vi'] # corpora callosa
Nsamples = vi.shape[1]
N = (vi.shape[0]/DIM-1)
qs = vi[0:-2,:].T.reshape([Nsamples,N,DIM])
qs = 1./50*qs # rescale
# reduce number of points
nrpts = 11
inds = np.linspace(0,N-N/nrpts,nrpts).astype('int')
qsfull = qs
qs = qs[:,inds,:]
N = qs.shape[1]
# reduce number of sampls
nrsamples = 10
inds = np.linspace(0,Nsamples-Nsamples/nrsamples,nrsamples).astype('int')
qsfull = qsfull[inds,:,:]
qs = qs[inds,:,:]
Nsamples = qs.shape[0]

#(mu,SigmaSQRT,lambdag,Logyis) = est.est(qs,SIGMA,WEIGHTS,rank,visualize=True,visualizeIterations=False,maxIter=maxIter)
## save output
#np.save('output/cc_est_mu',mu)
#np.save('output/cc_est_SigmaSQRT',SigmaSQRT)
#np.save('output/cc_est_lamdag',lambdag)
#np.save('output/cc_est_setup',[N,DIM,SIGMA,rank])
dir = '../../results/cc/11npts'
mu = np.load(dir+'/cc_est_mu.npy')
SigmaSQRT = np.load(dir+'/cc_est_SigmaSQRT.npy')
lambdag = np.load(dir+'/cc_est_lamdag.npy')
lres = np.load(dir+'/cc_est_setup.npy')
N = lres[0]; DIM = lres[1]; SIGMA = lres[2]; rank = lres[3];

# visualize
fixed=(mu, )
state0 = np.zeros(2*(N*DIM+N*DIM*rank))
state0[0:N*DIM] = mu.flatten()
state0[N*DIM:N*DIM+N*DIM*rank] = SigmaSQRT.flatten()
#stateT = y_span[-1,:]
#fixed=(stateT[0:N*DIM].reshape([N,DIM]), )

visualize=True
if visualize:
    #i = 0 # shape to visualize
    #i = 7 # shape to visualize
    i = 3 # shape to visualize (comment out dotted outline and point trajectories for this)
    print i

    moving=(qs[i,:,:], )

    #sim = ptsim.get(fixed, moving=moving, visualize=True)
    #sim['SIGMA'] = SIGMA
    #simT = sim['f'](state0, state0=state0, visualize=True, extra_points=qsfull[i,:,:])

    # match mu against a shape
    fixed=(mu, )
    sim = ptsim.get(fixed, visualize=True)
    sim['SIGMA'] = SIGMA
    Xa = SigmaSQRT

    extra_points = qsfull[i,:,:]
    (fstate,res) = match.match(sim,Xa,SIGMA,WEIGHTS,lambdag,initial=moving,visualize=True,visualizeIterations=False,maxIter=10, extra_points=extra_points)
    print fstate
    print res

#if True: # res.success:
#    match.genStateData(fstate,sim)
