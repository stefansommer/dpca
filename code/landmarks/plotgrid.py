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

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import det
import itertools

"""
Example usage:
plt.clf()
(grid,Nx,Ny)=pg.getGrid(-1,1,-1,1,xpts=50,ypts=50)
pg.plotGrid(grid,Nx,Ny)
"""


def d2zip(grid):
    return np.dstack(grid).reshape([-1,2])

def d2unzip(points,Nx,Ny):
    return np.array([points[:,0].reshape(Nx,Ny),points[:,1].reshape(Nx,Ny)])

def getGrid(xmin,xmax,ymin,ymax,xres=None,yres=None,xpts=None,ypts=None):
    """
    Make regular grid 
    Grid spacing is determined either by (x|y)res or (x|y)pts
    """

    if xres:
        xd = xres
    elif xpts:
        xd = np.complex(0,xpts)
    else:
        assert(False)
    if yres:
        yd = yres
    elif ypts:
        yd = np.complex(0,ypts)
    else:
        assert(False)

    grid = np.mgrid[xmin:xmax:xd,ymin:ymax:yd]
    Nx = grid.shape[1]
    Ny = grid.shape[2]

    return (d2zip(grid),Nx,Ny)
    

def plotGrid(grid,Nx,Ny,coloring=False):
    """
    Plot grid
    """

    xmin = grid[:,0].min(); xmax = grid[:,0].max()
    ymin = grid[:,1].min(); ymax = grid[:,1].max()
    border = .5*(0.2*(xmax-xmin)+0.2*(ymax-ymin))

    grid = d2unzip(grid,Nx,Ny)

    color = 0.75
    colorgrid = np.full([Nx,Ny],color)
    cm = plt.cm.get_cmap('gray')
    if coloring:
        cm = plt.cm.get_cmap('coolwarm')
        hx = (xmax-xmin) / (Nx-1)
        hy = (ymax-ymin) / (Ny-1)
        for i,j in itertools.product(range(Nx),range(Ny)):
            p = grid[:,i,j]
            xs = np.empty([0,2])
            ys = np.empty([0,2])
            if 0 < i:
                xs = np.vstack((xs,grid[:,i,j]-grid[:,i-1,j],))
            if i < Nx-1:
                xs = np.vstack((xs,grid[:,i+1,j]-grid[:,i,j],))
            if 0 < j:
                ys = np.vstack((ys,grid[:,i,j]-grid[:,i,j-1],))
            if j < Ny-1:
                ys = np.vstack((ys,grid[:,i,j+1]-grid[:,i,j],))

            Jx = np.mean(xs,0) / hx
            Jy = np.mean(ys,0) / hy
            J = np.vstack((Jx,Jy,)).T
            
            A = .5*(J+J.T)-np.eye(2)
            CSstrain = np.log(np.trace(A*A.T))
            logdetJac = np.log(det(J))
            colorgrid[i,j] = logdetJac

        cmin = np.min(colorgrid)
        cmax = np.max(colorgrid)
        f = 2*np.max((np.abs(cmin),np.abs(cmax),.5))
        colorgrid = colorgrid / f + 0.5

        print "mean color: %g" % np.mean(colorgrid)

    # plot lines
    for i,j in itertools.product(range(Nx),range(Ny)):
        if i < Nx-1:
            plt.plot(grid[0,i:i+2,j],grid[1,i:i+2,j],color=cm(colorgrid[i,j]))
        if j < Ny-1:
            plt.plot(grid[0,i,j:j+2],grid[1,i,j:j+2],color=cm(colorgrid[i,j]))

    #for i in range(0,grid.shape[1]):
    #    plt.plot(grid[0,i,:],grid[1,i,:],color)
    ## plot x lines
    #for i in range(0,grid.shape[2]):
    #    plt.plot(grid[0,:,i],grid[1,:,i],color)


    plt.xlim(xmin-border,xmax+border)
    plt.ylim(ymin-border,ymax+border)
 
