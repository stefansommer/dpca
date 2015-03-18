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
Perform mean and covariance estimation using supplied similarity measure using the mpps.
"""

import numpy as np
import mpp
from scipy.optimize import minimize,fmin_bfgs,fmin_cg,fmin_l_bfgs_b,root
from scipy.optimize import check_grad
from scipy.optimize import approx_fprime
from scipy import linalg
# from scipy.optimize import fmin_bfgs
import matplotlib.pyplot as plt
import itertools
import logging
from functools import partial

import dill
from pathos.multiprocessing import ProcessingPool
from pathos.multiprocessing import cpu_count
P = ProcessingPool(cpu_count()/2)

N_t = 100
t_span = np.linspace(0. ,2. , N_t )
N = None
DIM = None
rank = None
weights = None
ps = None


def getf():

    _N = N
    _DIM = DIM
    _rank = rank
    _SIGMA = mpp.SIGMA
    _ps = ps
    _weights = weights

    def f(m, full=False, onlyidx=None):

        # setup
        N = _N
        DIM = _DIM
        rank = _rank
        SIGMA = _SIGMA
        mpp.DIM = DIM
        mpp.N = N
        mpp.SIGMA = SIGMA
        mpp.rank = rank
        mpp.gaussian.N = N
        mpp.gaussian.DIM = DIM
        mpp.gaussian.SIGMA = mpp.SIGMA
        ps = _ps
        weights = _weights

        def MPPLogf((idx,m,lambdag)):
            #print "computing Log for sample %d" % idx
                    
            # setup
            N = _N
            DIM = _DIM
            rank = _rank
            SIGMA = _SIGMA
            mpp.DIM = DIM
            mpp.N = N
            mpp.SIGMA = SIGMA
            mpp.rank = rank
            mpp.gaussian.N = N
            mpp.gaussian.DIM = DIM
            mpp.gaussian.SIGMA = mpp.SIGMA
            ps = _ps
            weights = _weights

            # input
            mpp.lambdag = lambdag
            Nsamples = (m.shape[0]-(N*DIM+N*DIM*rank))/(N*DIM+N*DIM*rank)
            x0 = (1./Nsamples)*m[0:N*DIM]
            Xa0 = (1./Nsamples)*m[N*DIM:N*DIM+N*DIM*rank]
            Logsamples = m[N*DIM+N*DIM*rank:].reshape([-1,N*DIM+N*DIM*rank])
            xi0 = Logsamples[idx,0:N*DIM]
            xia0 = Logsamples[idx,N*DIM:N*DIM+N*DIM*rank]

            # flow
            state0 = mpp.weinstein_darboux_to_state( x0, Xa0, xi0, xia0 )
            x0,Xa0,xi0,xia0 = mpp.state_to_weinstein_darboux( state0 )
            (t_span, y_span) = mpp.integrate(state0)
            stateT = y_span[-1]
            xT,XaT,xiT,xiaT = mpp.state_to_weinstein_darboux( stateT )

            v0 = ps[idx,:,:]-xT.reshape([N,DIM])
            res = np.einsum('ia,ia',v0,v0) # 1./N ??
            #logging.info('match term after flow: ' + str(res))

            EH = mpp.Hamiltonian(x0,Xa0,xi0,xia0) # path energy from Hamiltonian
            #logging.info('Hamiltonian: ' + str(EH))

            #print "computed Log for sample %d (lambdag %g)" % (idx,mpp.lambdag)

            return weights[0]*EH+weights[1]*res


        # parallel compute distances        
        Nsamples = (m.shape[0]-(N*DIM+N*DIM*rank))/(N*DIM+N*DIM*rank)

        # determine lambdag
        #logging.info("determining lambdag...")
        x0 = (1./Nsamples)*m[0:N*DIM]
        Xa0 = (1./Nsamples)*m[N*DIM:N*DIM+N*DIM*rank].reshape([N*DIM,rank])
        (gsharp,_,_,g,_) = mpp.gs(x0)
        def detlambdag(lg):
            delta1 = np.eye(rank)
            W = np.einsum('ab,ka,lb->kl',delta1,Xa0,Xa0)+lg*gsharp
            W2 = np.einsum('ba,bi,ij->aj',W,g,W)
            detgsharp = np.linalg.det(gsharp)
            detW2 = np.linalg.det(W2)
            #print "detlambdag: lg %g, detW2 %g, detgsharp %g" % (lg,detW2,detgsharp)
            return detW2-detgsharp

        if rank > 0:
            reslambdag = root(detlambdag,1.)
            assert(reslambdag.success)
            lambdag = reslambdag.x
        else:
            lambdag = 1.
        
        # run logs
        #logging.info("performing shots...")
        input_args = zip(*(xrange(Nsamples), itertools.cycle((m,)), itertools.cycle((lambdag,)),))
        if onlyidx == None:
            sol = P.imap(MPPLogf, input_args)
            Logs = np.array(list(sol))            
            #Logs = np.empty(Nsamples)
            #for i in range(Nsamples):
            #   Logs[i] = MPPLogf(input_args[i])
        else:
            sampleid = onlyidx/(N*DIM+N*DIM*rank)
            logging.info("only idx %d, sample %d...",onlyidx,sampleid)
            Logs = np.zeros(Nsamples)
            Logs[sampleid] = MPPLogf(input_args[sampleid])

        res = (1./Nsamples)*np.sum(Logs)

        ## debug output
        if not full and onlyidx == None:
            #print "f x0: %s, Xa: %s, res %g" % (x0,Xa0,res,)
            print "f res %g" % (res,)

        if not full:
            return res
        else:
            return (res,(1./Nsamples)*Logs)

    return f

#def constr(m):
#    # constraints on frame
#    Nsamples = (m.shape[0]-(N*DIM+N*DIM*rank))/(N*DIM+N*DIM*rank)
#    x0 = (1./Nsamples)*m[0:N*DIM]
#    Xa0 = (1./Nsamples)*m[N*DIM:N*DIM+N*DIM*rank].reshape([N*DIM,N*DIM])
#    (_,_,_,gx0,_) = mpp.gs(x0)
#    Xa02inner = np.einsum('ba,bi,ij->aj',Xa0,gx0,Xa0)
#    detXa02 = np.linalg.det(Xa02inner)
#    
#    res = -np.sum(np.abs([1-detXa02]))
#    print "constr res: %s" % res
#    
#    return res

def err_func_gradient(p):

    logging.info("gradient...")

    f = getf()

    (fp,Logs) = f(p,full=True)
    #lsingle_grad_point = partial(single_grad_point, fp)

    _N = N
    _DIM = DIM
    _rank = rank
    _SIGMA = mpp.SIGMA
    _ps = ps
    _weights = weights
    lambdag = None

    def single_grad_point((idx,px,Logs)):
        # setup
        N = _N
        DIM = _DIM
        rank = _rank

        p = px.copy()
        epsilon = 1e-6
        p[idx] += epsilon
        if idx < N*DIM+N*DIM*rank:
            d1 = f(p)
            return (d1-fp)/(epsilon)
        else:
            onlyidx = idx-(N*DIM+N*DIM*rank)
            sampleid = onlyidx/(N*DIM+N*DIM*rank)
            d1 = f(p, onlyidx=onlyidx)
            return (d1-Logs[sampleid])/(epsilon)
        #p[idx] -= 2*epsilon
        #d2 = err_func(p)
        #return (d1-d2)/(2*epsilon)

    Nsamples = (p.shape[0]-(N*DIM+N*DIM*rank))/(N*DIM+N*DIM*rank)
    res = np.zeros(p.shape)
    # divide into two cases, x,Xa and remaining shots
    r0 = (0,N*DIM+N*DIM*rank)
    for i in range(r0[0],r0[1]): # run this serially
        res[i] = single_grad_point((i,p,None), )

    r1 = (N*DIM+N*DIM*rank,p.size)
    assert((r1[1]-r1[0])==Nsamples*(N*DIM+N*DIM*rank))
    input_args = zip(*(xrange(r1[0],r1[1]), itertools.cycle((p,)), itertools.cycle((Logs,))))
    sol = P.imap(single_grad_point, input_args)
    res[r1[0]:r1[1]] = np.array(list(sol))
    #for i in range(res2.size):
    #   res[r1[0]+i] = single_grad_point(input_args[i])

    return res


def est(_ps,SIGMA,_weights,_rank,maxIter=150, visualize=False, visualizeIterations=False, x0=None, Xa0=None):
    """
    Perform mean/cov estimation using the supplied similarity measure, mpps
    and scipy's optimizer. Currently no no derivative information is used based.

    Weights determines the split between energy (weights[0]) and match term (weights[1])
    """

    # number of samples
    global ps
    ps = _ps
    Nsamples = ps.shape[0]

    # set flow parameters
    global DIM,N,rank
    mpp.DIM = DIM = ps.shape[2]
    mpp.N = N = ps.shape[1]
    mpp.SIGMA = SIGMA
    mpp.rank = rank = _rank
    mpp.lambdag = 1.
    mpp.init()
    global weights
    weights = _weights

    logging.info("Estimation parameters: rank %d, N %d, Nsamples %d, weights %s, SIGMA %g, maxIter %d, visualize %s, visualizeIterations %s",mpp.rank,N,Nsamples,weights,SIGMA,maxIter,visualize,visualizeIterations)


    if x0 == None:
        # initial point
        x0 = np.mean(ps,0).flatten()
    if Xa0 == None:
        # initial frame
        pss = ps.reshape([Nsamples,N*DIM])
        (eigv,eigV) = np.linalg.eig(1./(Nsamples-1)*np.dot(pss.T,pss))
        inds = eigv>1e-4
        assert(np.sum(inds) >= rank)
        FrPCA = np.einsum('ij,j->ij',eigV[:,inds],np.sqrt(eigv[inds]))
        Xa0 = FrPCA.reshape([N*DIM,np.sum(inds)])[:,0:rank]

    logging.info("initial point/frame, x0: %s, Xa0: %s",x0,Xa0)

    initval = np.hstack( (Nsamples*x0,Nsamples*Xa0.flatten(),np.zeros((Nsamples,N*DIM+N*DIM*rank)).flatten(),) ).astype('double')
    tol = 1e-4
    # use COBYLA for constrainted optimization
    if maxIter > 0:
        f = getf()
        #logging.info("checking gradient...")
        #from scipy.optimize import approx_fprime
        #findiffgrad1 = approx_fprime(initval,f,1e-7)
        #findiffgrad2 = err_func_gradient(initval)
        #logging.info("gradient difference: %g",np.linalg.norm(findiffgrad1-findiffgrad2,np.inf))
        logging.info("running optimizer...")
        res = minimize(f, initval, method='CG',\
                       tol=tol,\
#                       constraints={'type': 'ineq', 'fun': constr},\
                       options={'disp': True, 'maxiter': maxIter},\
                       jac=err_func_gradient
                       )
        
        if not res.success:
            print "mean/covar optimization failed:\n%s" % res

        mu = (1./Nsamples)*res.x[0:N*DIM]
        SigmaSQRT = (1./Nsamples)*res.x[N*DIM:N*DIM+N*DIM*rank]
        Logyis = res.x[N*DIM+N*DIM*rank:].reshape([Nsamples,N*DIM+N*DIM*rank])
    else:
        logging.info("not running optimizer (maxIter 0)")
            
        # determine lambdag
        logging.info("determining lambdag...")
        m = initval
        x0 = (1./Nsamples)*m[0:N*DIM]
        Xa0 = (1./Nsamples)*m[N*DIM:N*DIM+N*DIM*rank].reshape([N*DIM,rank])
        (gsharp,_,_,g,_) = mpp.gs(x0)
        def detlambdag(lg):
            delta1 = np.eye(rank)
            W = np.einsum('ab,ka,lb->kl',delta1,Xa0,Xa0)+lg*gsharp
            W2 = np.einsum('ba,bi,ij->aj',W,g,W)
            detgsharp = np.linalg.det(gsharp)
            detW2 = np.linalg.det(W2)
            #print "detlambdag: lg %g, detW2 %g, detgsharp %g" % (lg,detW2,detgsharp)
            return detW2-detgsharp

        if rank > 0:
            reslambdag = root(detlambdag,1.)
            assert(reslambdag.success)
            mpp.lambdag = reslambdag.x
        else:
            mpp.lambdag = 1.

        mu = (1./Nsamples)*initval[0:N*DIM]
        SigmaSQRT = (1./Nsamples)*initval[N*DIM:N*DIM+N*DIM*rank]
        Logyis = initval[N*DIM+N*DIM*rank:].reshape([Nsamples,N*DIM+N*DIM*rank])

    mu = mu.reshape([N,DIM])
    SigmaSQRT = SigmaSQRT.reshape([N*DIM,rank])
    print "mu: %s,\nSigmaSQRT: %s" % (mu,SigmaSQRT)
    print "diff %s" % np.linalg.norm(initval-res.x,np.inf)

    return (mu,SigmaSQRT,mpp.lambdag,Logyis)


def genStateData(fstate, sim):
    logging.info("generating state data for optimization result")

    mpp.DIM = DIM = sim['DIM']
    mpp.N = N = sim['N']
    mpp.SIGMA = SIGMA = sim['SIGMA']
    mpp.init()
    
    (t_span, y_span) = mpp.integrate( fstate )
    
    # save result
    np.save('output/est_final_fstate',fstate)
    np.save('output/est_setup',[N,DIM,SIGMA])
