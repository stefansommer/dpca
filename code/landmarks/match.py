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
Perform matching using supplied similarity measure using the mpps.
Please look in matching/ for relevant similarity measures, 
e.g. image matching and point matching.
"""

import numpy as np
import mpp
from scipy.optimize import minimize,fmin_bfgs,fmin_cg,fmin_l_bfgs_b
from scipy.optimize import check_grad
from scipy.optimize import approx_fprime
# from scipy.optimize import fmin_bfgs
import matplotlib.pyplot as plt
import itertools
import logging

N_t = 100
t_span = np.linspace(0. ,2. , N_t )

def F(sim, nonmoving, x, weights=None, visualize=False, extra_points=None):
    """
    Function that scipy's optimize function will call for returning the value 
    and gradient for a given x. The forward and adjoint integration is called 
    from this function using the values supplied by the similarity measure.
    """

    N = sim['N']
    DIM = sim['DIM']

    i = 0
    q = np.reshape( nonmoving[i:(i+N*DIM)] , [N,DIM] )
    mpp.gaussian.N = N
    mpp.gaussian.DIM = DIM
    mpp.gaussian.SIGMA = mpp.SIGMA

    # input
    state0 = np.append(nonmoving, x)
    x,Xa,xi,xia = mpp.state_to_weinstein_darboux( state0 )
    xia = 1./10*xia # hack to improve convergence
    state0 = mpp.weinstein_darboux_to_state(x,Xa,xi,xia)

    # flow
    (t_span, y_span) = mpp.integrate(state0)
    stateT = y_span[-1]
    
    simT = sim['f'](stateT, state0=state0, visualize=visualize, extra_points=extra_points)

    # debug
    logging.info('match term after flow: ' + str(simT[0]))
    
    Ediff = mpp.Hamiltonian(x,Xa,xi,xia) # path energy from Hamiltonian
    logging.info('Hamiltonian: ' + str(Ediff))

    return weights[1]*simT[0]+weights[0]*Ediff



def match(sim,Xa,SIGMA,weights,lambdag=None, initial=None, maxIter=150, visualize=False, visualizeIterations=False, extra_points=None):
    """
    Perform matching using the supplied similarity measure, mpps
    and scipy's optimizer. Currently no no derivative information is used based.

    The initial value is either zero of supplied as a parameter.

    Weights determines the split between energy (weights[0]) and match term (weights[1])
    """

    # set flow parameters
    DIM = mpp.DIM = sim['DIM']
    N = mpp.N = sim['N']
    mpp.SIGMA = SIGMA
    mpp.rank = Xa.shape[1]
    if mpp.rank < N*DIM:
        assert(lambdag)
        mpp.lambdag = lambdag
    mpp.init()

    logging.info("Flow parameters (rank %d): weights %s, Xa %s, SIGMA %g, lambdag %s, maxIter %d, visualize %s, visualizeIterations %s",mpp.rank,weights,Xa,SIGMA,lambdag,maxIter,visualize,visualizeIterations)

    # initial guess (x0moving)
    x = initial[0].flatten()
    xi = np.zeros(N*DIM)
    xia = np.zeros([N*DIM,mpp.rank])
    x0 = mpp.weinstein_darboux_to_state( x, Xa, xi, xia )

    sizeqs = mpp.get_dim_state()
    x0nonmoving = x0[0:sizeqs/2] # for now, the point positions and frame are fixed
    x0moving = x0[sizeqs/2:sizeqs] 

    # optimization functions
    fsim = lambda x: F(sim, x0nonmoving, x, weights=weights, visualize=visualizeIterations)

    # change optimization method to e.g. BFGS after debugging
    res = minimize(fsim, x0moving, method='CG', jac=False, options={'disp': True, 'maxiter': maxIter})
    resx = res.x

    # visualize result
    if visualize:
        F(sim, x0nonmoving, resx, weights=weights, visualize=True, extra_points=extra_points)

    return (np.append(x0nonmoving,res.x),res)


def genStateData(fstate, sim):
    logging.info("generating state data for optimization result")

    mpp.DIM = DIM = sim['DIM']
    mpp.N = N = sim['N']
    mpp.SIGMA = SIGMA = sim['SIGMA']
    mpp.init()
    
    (t_span, y_span) = mpp.integrate( fstate )
    
    # save result
    np.save('output/state_data',y_span)
    np.save('output/time_data',t_span)
    np.save('output/setup',[N,DIM,SIGMA])
