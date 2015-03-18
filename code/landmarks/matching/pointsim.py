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
Similarity measure for point matching.
Sum of squared distances applied to all values.
"""

import numpy as np
import mpp
from scipy import ndimage
import matplotlib.pyplot as plt
from numpy import sqrt
from functools import partial
import plotgrid as pg
import os
import itertools

def psim( state, N=None, DIM=None, rank=None, fixed=None, moving=None, visualize=False, state0=None, grid=None, extra_points=None):
    x,Xa,xi,xia = mpp.state_to_weinstein_darboux( state )
    qm = x.reshape([N,DIM])
    qf = fixed[0]

    # value
    v0 = qm-qf
    m0 = np.einsum('ia,ia',v0,v0) # 1./N ??
    
    # gradient
    dq0 = 2.*v0 # 1./N ??

    #print "point sim: m0 " + str(m0) + ", m1 " + str(m1) + ", m2 " + str(m2)

    ## visualization
    if visualize:
        plt.figure(1)
        plt.clf()

        if state0 != None:
            # grid
            if grid != None:
                (reggrid,Nx,Ny) = grid
                (t_span,y_span,mgridts) = mpp.integrate(state0,pts=reggrid)
                mgridT = mgridts[-1:].reshape(-1,DIM)
                pg.plotGrid(mgridT,Nx,Ny,coloring=True)
            else:
                (t_span,y_span) = mpp.integrate(state0)
                
            # plot curves and frames
            Q = y_span[:,0:N*DIM].reshape([-1,N,DIM])
            #for i in range(N): # comment out for mean
            #    plt.plot(Q[:,i,0],Q[:,i,1],'k-')
            for i in range(len(t_span)):
                x_i,Xa_i,xi_i,xia_i = mpp.state_to_weinstein_darboux( y_span[i,:] )
                x_i = x_i.reshape([N,DIM])
                Xa_i = Xa_i.reshape([N*DIM,mpp.rank])
                # plot frame
                colors = plt.get_cmap()(np.linspace(0,1,mpp.rank))
                #if i % 4 == 0 or i == len(t_span)-1: # plot every 4th frame
                if i == len(t_span)-1: # plot last frame
                    for j in range(mpp.rank):
                        Fr = Xa_i[:,j].reshape(N,DIM)
                        for k in range(N):
                            plt.quiver(x_i[k,0],x_i[k,1],Fr[k,0],Fr[k,1],color=colors[j],angles='xy', scale_units='xy', scale=3, width=.003) # scale=3 for paper plots

        if state0 != None and extra_points != None:
            (t_span,y_span,eps) = mpp.integrate(state0,pts=extra_points)
            eps = eps[-1:].reshape(-1,DIM)
            # for mean:
            #plt.plot(extra_points[:,0],extra_points[:,1],'--',color='gray')
            plt.plot(eps[:,0],eps[:,1],'gray')
            ## for ccs:
            #plt.plot(extra_points[:,0],extra_points[:,1],color='gray')
            #plt.plot(eps[:,0],eps[:,1],'--',color='gray')

        plt.plot(qf[:,0],qf[:,1],'bo')
        plt.plot(qm[:,0],qm[:,1],'ro')
        if moving != None:
            qq = moving[0]
            plt.plot(qq[:,0],qq[:,1],'go')

        border = 0.6 # .6 for paper plots
        plt.axis('equal')
        plt.xlim(min(np.vstack((qf,qm))[:,0])-border,max(np.vstack((qf,qm))[:,0])+border)
        plt.ylim(min(np.vstack((qf,qm))[:,1])-border,max(np.vstack((qf,qm))[:,1])+border)
        plt.axis('off')
        plt.draw()
        #plt.show(block=False)

        # save figures
        for i in plt.get_fignums():
            plt.figure(i)
            try:
                os.mkdir('output/%s' % os.getpid() )
            except:
                None
            plt.savefig('output/%s/figure%d.eps' % (os.getpid(),i) )

    return (m0, (dq0, ))

def get(fixed=None, visualize=False, moving=None):
    """
    get point SSD similarity 
    """

    # data
    qf = fixed[0]

    N = qf.shape[0]
    DIM = qf.shape[1]
    
    ## visualization
    reggrid = None
    if visualize:
        if moving: # moving points needed in order to find grid size
            qm = moving[0]
            reggrid = pg.getGrid(np.vstack((qf,qm))[:,0].min()-1,np.vstack((qf,qm))[:,0].max()+1,np.vstack((qf,qm))[:,1].min()-1,np.vstack((qf,qm))[:,1].max()+1,xpts=40,ypts=40)
            #pg.plotGrid(*reggrid,coloring=True)

    f = partial(psim, fixed=fixed, moving=moving, N=N, DIM=DIM, visualize=visualize, grid=reggrid)
    sim = {'f': f, 'N': N, 'DIM': DIM}

    return sim
    
