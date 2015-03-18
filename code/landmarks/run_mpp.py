#!/usr/bin/python

import mpp
import numpy as np
from scipy import linalg
import matching.pointsim as ptsim

DIM = mpp.DIM = 2
N = mpp.N = 3
SIGMA = mpp.SIGMA = 1.
mpp.rank = N*DIM
mpp.init()

np.random.seed(100)

def d2zip(grid):
    return np.dstack(grid).reshape([-1,2])

x = np.array([[-1.0,0.0],[1.0,0.0],[0.0,0.00]])
xi = np.array([[.5,2.0],[.5,2.0],[.5,2.0]])
x = x[0:N,:]
xi = xi[0:N,:]

# specify frame/rank d map
(gsharp,_,_,g,_) = mpp.gs(x)
#Xa = np.outer(np.ones(N),np.eye(DIM)).reshape([N,DIM,DIM])

# length of frame vectors (stds) in gsharp coordinates and covariances
stds = np.array([1,1]) # more freedom in x direction  (2,.5 previously)
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

xia = np.zeros([N*DIM,mpp.rank])

mpp.test_functions(1)
(t_span, y_span) = mpp.integrate(mpp.weinstein_darboux_to_state(x,Xa,xi,xia), T=1.)

print 'initial energy was \n' + str(mpp.energy(y_span[0]))
print 'final energy is \n'    + str(mpp.energy(y_span[-1]))

# visualize
moving=(x, )
state0 = y_span[0,:]
stateT = y_span[-1,:]
fixed=(stateT[0:N*DIM].reshape([N,DIM]), )

visualize=True
if visualize:
    sim = ptsim.get(fixed, moving=moving, visualize=True)
    sim['SIGMA'] = SIGMA
    simT = sim['f'](stateT, state0=state0, visualize=True)

# save result
np.save('state_data',y_span)
np.save('time_data',t_span)
np.save('setup',[N,DIM,SIGMA])
