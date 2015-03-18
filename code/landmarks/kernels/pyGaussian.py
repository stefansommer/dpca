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
    Wrapper for gaussian.py
"""

import numpy as np
import gaussian

N = None
DIM = None
SIGMA = None

def Gaussian_monomial( x , n ):
# computes x/sigma^n * G(x)
    y = x / SIGMA
    store = y * np.exp( -(0.5/n) * y**2 )
    return store**n

def diff_1D_Gaussian_cpp(x,k,SIGMA,parallel=False):
    s = x.shape
    out = np.zeros(x.size)
    gaussian.diff_1D_Gaussian_parallel_cpp(x.flatten(),out,k,SIGMA,parallel)
    return out.reshape(s)

def diff_1D_Gaussian( x , k ):
# returns the kth derivative of a 1 dimensional Guassian
    G = np.exp( -0.5 * (x / SIGMA)**2 )
    if k == 0:
        return G
    elif k==1:
        return -1.*Gaussian_monomial(x,1) / (SIGMA)
    elif k==2:
        return ( Gaussian_monomial(x,2) - G ) / (SIGMA**2)
    elif k==3:
        return -1.*( Gaussian_monomial(x,3) - 3.*Gaussian_monomial(x,1)) / (SIGMA**3)
    elif k==4:
        return (Gaussian_monomial(x,4) - 6.*Gaussian_monomial(x,2) + 3.*G ) / (SIGMA**4)
    elif k==5:
        return (-1.*(Gaussian_monomial(x,5) - 10.*Gaussian_monomial(x,3) + 15.*Gaussian_monomial(x,1) ))/(SIGMA**5)
    elif k==6:
        return (Gaussian_monomial(x,6) - 15.*Gaussian_monomial(x,4) + 45.*Gaussian_monomial(x,2) -15.*G)/(SIGMA**6)
    else:
        print 'error in diff_1D_Guassian:  k='+str(k)
        return 'error'

def derivatives_of_Gaussians( p1 , p2, parallel=False ):
    N_p1 = p1.shape[0]
    N_p2 = p2.shape[0]
    r_sq = np.zeros( [ N_p1 , N_p2 ] )
    dx = np.zeros( [N_p1,N_p2,DIM] )
    for a in range(0,DIM):
        dx[:,:,a] = np.outer( p1[:,a] , np.ones(N_p2) ) - np.outer( np.ones(N_p1), p2[:,a] )
        r_sq[:,:] = dx[:,:,a]**2 + r_sq[:,:]
    G = np.exp( - r_sq / (2.*SIGMA**2) )
    DG = np.ones( [N_p1,N_p2,DIM] )
    D2G = np.ones( [N_p1,N_p2,DIM,DIM] )
    D3G = np.ones( [N_p1,N_p2,DIM,DIM,DIM] )
    D4G = np.ones( [N_p1,N_p2,DIM,DIM,DIM,DIM] )
    D5G = np.ones( [N_p1,N_p2,DIM,DIM,DIM,DIM,DIM] )
    D6G = np.ones( [N_p1,N_p2,DIM,DIM,DIM,DIM,DIM,DIM] )
    alpha = np.int_(np.zeros(DIM))
    #one derivative
    for a in range(0,DIM):
        alpha[a] = 1
        for b in range(0,DIM):
            #DG[:,:,a] = DG[:,:,a]*diff_1D_Gaussian( dx[:,:,b] , alpha[b] )
            DG[:,:,a] = DG[:,:,a]*diff_1D_Gaussian_cpp( dx[:,:,b] , alpha[b], SIGMA, parallel )
        alpha[a] = 0
    
    #two derivatives
    for a in range(0,DIM):
        alpha[a] = 1
        for b in range(0,DIM):
            alpha[b] = alpha[b] + 1
            for c in range(0,DIM):
                #D2G[:,:,a,b] = D2G[:,:,a,b]*diff_1D_Gaussian( dx[:,:,c] , alpha[c] )
                D2G[:,:,a,b] = D2G[:,:,a,b]*diff_1D_Gaussian_cpp( dx[:,:,c] , alpha[c], SIGMA, parallel )
            alpha[b] = alpha[b] - 1
        alpha[a] = 0

    #three derivatives
    for a in range(0,DIM):
        alpha[a] = 1
        for b in range(0,DIM):
            alpha[b] = alpha[b] + 1
            for c in range(0,DIM):
                alpha[c] = alpha[c] + 1
                for d in range(0,DIM):
                    #D3G[:,:,a,b,c] = D3G[:,:,a,b,c]*diff_1D_Gaussian( dx[:,:,d] , alpha[d] )
                    D3G[:,:,a,b,c] = D3G[:,:,a,b,c]*diff_1D_Gaussian_cpp( dx[:,:,d] , alpha[d], SIGMA, parallel )
                alpha[c] = alpha[c] - 1
            alpha[b] = alpha[b] - 1
        alpha[a] = 0

    #four derivatives
    for a in range(0,DIM):
        alpha[a] = 1
        for b in range(0,DIM):
            alpha[b] = alpha[b] + 1
            for c in range(0,DIM):
                alpha[c] = alpha[c] + 1
                for d in range(0,DIM):
                    alpha[d] = alpha[d] + 1
                    for e in range(0,DIM):
                        #D4G[:,:,a,b,c,d] = D4G[:,:,a,b,c,d]*diff_1D_Gaussian( dx[:,:,e] , alpha[e] )
                        D4G[:,:,a,b,c,d] = D4G[:,:,a,b,c,d]*diff_1D_Gaussian_cpp( dx[:,:,e] , alpha[e], SIGMA, parallel )
                    alpha[d] = alpha[d] - 1
                alpha[c] = alpha[c] - 1
            alpha[b] = alpha[b] - 1
        alpha[a] = 0

    #five derivatives
    for a in range(0,DIM):
        alpha[a] = 1
        for b in range(0,DIM):
            alpha[b] = alpha[b] + 1
            for c in range(0,DIM):
                alpha[c] = alpha[c] + 1
                for d in range(0,DIM):
                    alpha[d] = alpha[d] + 1
                    for e in range(0,DIM):
                        alpha[e] = alpha[e] + 1
                        for f in range(0,DIM):
                            #D5G[:,:,a,b,c,d,e] = D5G[:,:,a,b,c,d,e]*diff_1D_Gaussian( dx[:,:,f] , alpha[f] )
                            D5G[:,:,a,b,c,d,e] = D5G[:,:,a,b,c,d,e]*diff_1D_Gaussian_cpp( dx[:,:,f] , alpha[f], SIGMA, parallel )
                        alpha[e] = alpha[e] - 1
                    alpha[d] = alpha[d] - 1
                alpha[c] = alpha[c] - 1
            alpha[b] = alpha[b] - 1
        alpha[a] = 0

    #six derivatives
    for a in range(0,DIM):
        alpha[a] = 1
        for b in range(0,DIM):
            alpha[b] = alpha[b] + 1
            for c in range(0,DIM):
                alpha[c] = alpha[c] + 1
                for d in range(0,DIM):
                    alpha[d] = alpha[d] + 1
                    for e in range(0,DIM):
                        alpha[e] = alpha[e] + 1
                        for f in range(0,DIM):
                            alpha[f] = alpha[f] + 1
                            for g in range(0,DIM):
                                #D6G[:,:,a,b,c,d,e,f] = D6G[:,:,a,b,c,d,e,f]*diff_1D_Gaussian( dx[:,:,g] , alpha[g] )
                                D6G[:,:,a,b,c,d,e,f] = D6G[:,:,a,b,c,d,e,f]*diff_1D_Gaussian_cpp( dx[:,:,g] , alpha[g], SIGMA, parallel )
                            alpha[f] = alpha[f] - 1
                        alpha[e] = alpha[e] - 1
                    alpha[d] = alpha[d] - 1
                alpha[c] = alpha[c] - 1
            alpha[b] = alpha[b] - 1
        alpha[a] = 0
    return G, DG, D2G, D3G, D4G, D5G , D6G

