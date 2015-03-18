#from scipy.spatial.distance import pdist , squareform
import numpy as np
from scipy.integrate import odeint
import kernels.pyGaussian as gaussian
import multiprocessing as mp
from functools import partial
import ctypes
import time
import itertools
import logging

import __builtin__
try:
    __debug = __builtin__.__debug
except AttributeError:
    __debug = False

N = None
DIM = None
SIGMA = None

parallel = False
pool = None
tic = None
nrProcesses = None

# use rank d scheme if lower than N*DIM
rank = None
lambdag = 0. # weight on Riemannian metric for low rank scheme
#rank = 1

def getN():
    return N

def getDIM():
    return DIM

def getrank():
    return rank

def get_dim_state():
    return 2*(N*DIM+N*DIM*rank)

def isFullRank():
    fullRank = rank == N*DIM
    assert(fullRank != (lambdag > 0.))
    return fullRank

def derivatives_of_kernel( nodes , x, ompParallel=parallel ):
    #given x_i and x_j the K = Kernel( x_ij) and derivatives with x_ij = x_i - x_j.
    #The code is written such that we evaluate at the nodes, and entry (i,j) is the contribution at node i due to particle j.
    delta = np.identity( DIM )
    G,DG,D2G,D3G,D4G,D5G,D6G = gaussian.derivatives_of_Gaussians( nodes.reshape([-1,DIM]) , x.reshape([N,DIM]), ompParallel )
    K = np.einsum('ij,ab->ijab',G,delta)
    DK = np.einsum('ijc,ab->ijabc',DG,delta)
    D2K = np.einsum('ijcd,ab->ijabcd',D2G,delta)
    #D3K = np.einsum('ijcde,ab->ijabcde',D3G,delta)
    #EXAMPLE OF INDEX CONVENTION 'ijabc' refers to the c^th derivative of the ab^th entry of K(q_i - q_j)
    return K, DK, D2K

def Gamma_g(gsharp,Dgsharp,D2gsharp,g,Dg):
    """ Christoffel symbols from metric """

    T = ( np.einsum('kl,rsl->krs',gsharp,Dgsharp)\
            -np.einsum('sl,rkl->krs',gsharp,Dgsharp)\
            -np.einsum('rl,ksl->krs',gsharp,Dgsharp)\
            )
    Gamma = .5*np.einsum('ir,krs,sj->kij',g,T,g)

    return Gamma

def DGamma_g(gsharp,Dgsharp,D2gsharp,g,Dg):
    """ derivative of Christoffel symbols """

    T = ( np.einsum('kl,rsl->krs',gsharp,Dgsharp)\
            -np.einsum('sl,rkl->krs',gsharp,Dgsharp)\
            -np.einsum('rl,ksl->krs',gsharp,Dgsharp)\
            )
    DT = ( np.einsum('klx,rsl->krsx',Dgsharp,Dgsharp)\
            +np.einsum('kl,rslx->krsx',gsharp,D2gsharp)\
            -np.einsum('slx,rkl->krsx',Dgsharp,Dgsharp)\
            -np.einsum('sl,rklx->krsx',gsharp,D2gsharp)\
            -np.einsum('rlx,ksl->krsx',Dgsharp,Dgsharp)\
            -np.einsum('rl,kslx->krsx',gsharp,D2gsharp)\
            )
    DGamma = .5*(np.einsum('irx,krs,sj->kijx',Dg,T,g)\
                 +np.einsum('ir,krsx,sj->kijx',g,DT,g)\
                 +np.einsum('ir,krs,sjx->kijx',g,T,Dg)\
                 )

    return DGamma

def gs(x):
    """ return cometric, metric, and derivatives """
    K,DK,D2K = derivatives_of_kernel(x,x)

    delta = np.identity( N )
    ones = np.ones( N )

    # metric is in 'iajb' format, shape [N*DIM,N*DIM]
    gsharp = np.einsum('ijab->iajb',K).reshape([N*DIM,N*DIM]) # cometric
    deltasubdelta = np.einsum('il,j->ilj',delta,ones)-np.einsum('ij,l->ilj',delta,ones)
    Dgsharp = np.einsum('ljcba,ilj->lcjbia',DK,deltasubdelta).reshape([N*DIM,N*DIM,N*DIM])
    D2gsharp = np.einsum('ljcbax,ilj,klj->lcjbiakx',D2K,deltasubdelta,deltasubdelta).reshape([N*DIM,N*DIM,N*DIM,N*DIM])

    # invert metric
    g = np.linalg.inv(gsharp)
    Dg = -np.einsum('jr,rsk,sm->jmk',g,Dgsharp,g)

    return (gsharp,Dgsharp,D2gsharp,g,Dg)

def Hamiltonian( x, Xa, xi, xia ):
    #returns the Hamiltonian.  Serves as a safety to check our equations of motion are correct.
    delta1 = np.eye(rank)

    gsharp,Dgsharp,D2gsharp,g,Dg = gs(x)
    Gamma = Gamma_g(gsharp,Dgsharp,D2gsharp,g,Dg)
    GammaX = np.einsum('hja,ag->hgj',Gamma,Xa)

    W = np.einsum('ab,ka,lb->kl',delta1,Xa,Xa)+lambdag*gsharp
    dx = np.einsum('ij,j->i',W,xi)-np.einsum('ih,jbh,jb->i',W,GammaX,xia)
    dXa = -np.einsum('iah,hj,j->ia',GammaX,W,xi)+np.einsum('iak,kh,jbh,jb->ia',GammaX,W,GammaX,xia)
        
    return .5*(np.einsum('i,i->',dx,xi)+np.einsum('ia,ia->',dXa,xia))
    
def ode_function( x , t ):
    #print "ode_function, t: " + str(t)

    state = x[0:get_dim_state()]
    pts = x[get_dim_state():].reshape(-1,DIM)

    x, Xa, xi, xia = state_to_weinstein_darboux( state )

    # get (co-)metric and Christoffel symbols with derivatives
    gsharp,Dgsharp,D2gsharp,g,Dg = gs(x)
    Gamma = Gamma_g(gsharp,Dgsharp,D2gsharp,g,Dg)
    d0Gamma = DGamma_g(gsharp,Dgsharp,D2gsharp,g,Dg)
    delta0 = np.eye(N*DIM) # for coordinates
    delta1 = np.eye(rank) # for frames/lower rank maps
    
    GammaX = np.einsum('hja,ag->hgj',Gamma,Xa)
    d0GammaX = np.einsum('hjak,ag->hgjk',d0Gamma,Xa)        
    d1GammaX = np.einsum('xa,ihl->iahlx',delta1,Gamma)
    
    W = np.einsum('ab,ka,lb->kl',delta1,Xa,Xa)+lambdag*gsharp
    d0W = lambdag*Dgsharp
    d1W = np.einsum('il,jx->ijlx',delta0,Xa)+np.einsum('jl,ix->ijlx',delta0,Xa)
    
    d0g00 = d0W
    d0g01 = -np.einsum('ihl,jbh->ijbl',d0W,GammaX)-np.einsum('ih,jbhl->ijbl',W,d0GammaX)
    d0g10 = -np.einsum('iahl,hj->iajl',d0GammaX,W)-np.einsum('iah,hjl->iajl',GammaX,d0W)
    d0g11 = (np.einsum('iakl,kh,jbh->iajbl',d0GammaX,W,GammaX)+\
             np.einsum('iak,khl,jbh->iajbl',GammaX,d0W,GammaX)+\
             np.einsum('iak,kh,jbhl->iajbl',GammaX,W,d0GammaX))
    
    d1g00 = d1W
    d1g01 = -np.einsum('ihlx,jbh->ijblx',d1W,GammaX)-np.einsum('ih,jbhlx->ijblx',W,d1GammaX)
    d1g10 = -np.einsum('iahlx,hj->iajlx',d1GammaX,W)-np.einsum('iah,hjlx->iajlx',GammaX,d1W)
    d1g11 = (np.einsum('iaklx,kh,jbh->iajblx',d1GammaX,W,GammaX)+\
             np.einsum('iak,khlx,jbh->iajblx',GammaX,d1W,GammaX)+\
             np.einsum('iak,kh,jbhlx->iajblx',GammaX,W,d1GammaX))
    
    dx = np.einsum('ij,j->i',W,xi)-np.einsum('ih,jbh,jb->i',W,GammaX,xia)
    dXa = -np.einsum('iah,hj,j->ia',GammaX,W,xi)+np.einsum('iak,kh,jbh,jb->ia',GammaX,W,GammaX,xia)
    dxi = -.5*(np.einsum('hki,h,k->i',d0g00,xi,xi)+\
               np.einsum('hkdi,h,kd->i',d0g01,xi,xia)+\
               np.einsum('hgki,hg,k->i',d0g10,xia,xi)+\
               np.einsum('hgkdi,hg,kd->i',d0g11,xia,xia))
    dxia = -.5*(np.einsum('hkia,h,k->ia',d1g00,xi,xi)+\
                np.einsum('hkdia,h,kd->ia',d1g01,xi,xia)+\
                np.einsum('hgkia,hg,k->ia',d1g10,xia,xi)+\
                np.einsum('hgkdia,hg,kd->ia',d1g11,xia,xia))

    dstate = weinstein_darboux_to_state( dx, dXa, dxi, dxia )

    # points carried along the flow (using horizontal lift w.r.t. g)
    K,_,_ = derivatives_of_kernel( pts , x )
    p = np.einsum('ij,j->i',g,dx)
    dpts = np.einsum('ijab,jb->ia',K,p.reshape([N,DIM]))

    return np.hstack((dstate,dpts.flatten()))

#def flow_points(state, s, pts):
#    # points carried along the flow
#    x, Xa, xi, xia = state_to_weinstein_darboux( state )
#    K,DK,D2K,D3K,D4K,D5K,D6K = derivatives_of_kernel( pts , x )
#    p = g_flat(dx)
#    dpts = np.einsum('ijab,jb->ia',K,p)
#
#    return (s, dpts)

def state_to_weinstein_darboux( state, N=None, DIM=None, rank=None ):
    if N == None: N = getN()
    if DIM == None: DIM = getDIM()
    if rank == None: rank = getrank()

    i = 0
    x = np.reshape( state[i:(i+N*DIM)] , [N*DIM] )
    i = i + N*DIM
    Xa = np.reshape( state[i:(i+N*DIM*rank)] , [N*DIM,rank] )
    i = i + N*DIM*rank
    xi = np.reshape( state[i:(i+N*DIM)] , [N*DIM] )
    i = i + N*DIM
    xia = np.reshape( state[i:(i+N*DIM*rank)] , [N*DIM,rank] )
    return x, Xa , xi, xia

def weinstein_darboux_to_state( x, Xa, xi, xia , N=None , DIM=None, rank=None ):
    if N == None: N = getN()
    if DIM == None: DIM = getDIM()
    if rank == None: rank = getrank()

    state = np.zeros( get_dim_state() )
    i = 0
    state[i:(i+N*DIM)] = np.reshape( x , N*DIM )
    i = i + N*DIM
    state[i:(i + N*DIM*rank)] = np.reshape( Xa , N*DIM*rank )
    i = i + N*DIM*rank
    state[i:(i+N*DIM)] = np.reshape( xi , N*DIM )
    i = i + N*DIM
    state[i:(i + N*DIM*rank)] = np.reshape( xia , N*DIM*rank )

    return state


def init():
    assert(N)
    assert(DIM)
    assert(0 <= rank <= N*DIM)
    gaussian.N = N
    gaussian.DIM = DIM
    gaussian.SIGMA = SIGMA

def test_kernel_functions( q ):
    h = 1e-8
#    G,DG,D2G,D3G,D4G,D5G = gaussian.derivatives_of_Gaussians(q,q)
    q_a = np.copy(q)
    q_b = np.copy(q)
    q_c = np.copy(q)
    q_d = np.copy(q)
    q_e = np.copy(q)
    K,DK,D2K = derivatives_of_kernel(q,q)
    delta = np.identity(DIM)
    error_max = 0.
    for i in range(0,N):
        for j in range(0,N):
            x = q[i,:] - q[j,:]
            r_sq = np.inner( x , x )
            for a in range(0,DIM):
                for b in range(0,DIM):
                    G = np.exp( -r_sq / (2.*SIGMA**2) )
                    K_ij_ab = G*delta[a,b]
                    error = K_ij_ab - K[i,j,a,b]
                    error_max = np.maximum( np.absolute(error) , error_max )

    print 'error_max for K = ' + str( error_max )
    if (error_max > 100*h):
        print 'WARNING:  COMPUTATION OF K APPEARS TO BE INACCURATE'

    error_max = 0.
    for i in range(0,N):
        for a in range(0,DIM):
            q_a[i,a] = q[i,a] + h
            K_a,DK_a,D2K_a = derivatives_of_kernel(q_a,q)
            for j in range(0,N):
                der = ( K_a[i,j,:,:] - K[i,j,:,:] ) / h
                error = np.linalg.norm(  der - DK[i,j,:,:,a] )
                error_max = np.maximum(error, error_max)
            q_a[i,a] = q[i,a]
    print 'error_max for DK = ' + str( error_max )
    if (error_max > 100*h):
        print 'WARNING:  COMPUTATION OF DK APPEARS TO BE INACCURATE'

    error_max = 0.
    q_b = np.copy(q)
    for i in range(0,N):
        for a in range(0,DIM):
            for b in range(0,DIM):
                q_b[i,b] = q[i,b] + h
                K_b,DK_b,D2K_b = derivatives_of_kernel(q_b,q)
                for j in range(0,N):
                    der = (DK_b[i,j,:,:,a] - DK[i,j,:,:,a] )/h
                    error = np.linalg.norm( der - D2K[i,j,:,:,a,b] )
                    error_max = np.maximum( error, error_max )
                q_b[i,b] = q[i,b]

    print 'error_max for D2K = ' + str( error_max )
    if (error_max > 100*h):
        print 'WARNING:  COMPUTATION OF D2K APPEARS TO BE INACCURATE'

#    error_max = 0.
#    q_c = np.copy(q)
#    for i in range(0,N):
#        for a in range(0,DIM):
#            for b in range(0,DIM):
#                for c in range(0,DIM):
#                    q_c[i,c] = q[i,c] + h
#                    K_c,DK_c,D2K_c = derivatives_of_kernel(q_c,q)
#                    for j in range(0,N):
#                        der = (D2K_c[i,j,:,:,a,b] - D2K[i,j,:,:,a,b] )/h
#                        error = np.linalg.norm( der - D3K[i,j,:,:,a,b,c] )
#                        error_max = np.maximum( error, error_max )
#                    q_c[i,c] = q[i,c]
#
#    print 'error_max for D3K = ' + str( error_max )

    if (error_max > 100*h):
        print 'WARNING:  COMPUTATION OF D3K APPEARS TO BE INACCURATE'

    print 'TESTING SYMMETRIES'
    print 'Is K symmetric with respect to ij?'
    error_max = 0
    for i in range(0,N):
        for j in range(0,N):
            error = np.linalg.norm( K[i,j,:,:] - K[j,i,:,:] )
            error_max = np.maximum( error, error_max )
    print 'max for K_ij - K_ji = ' + str( error_max )

    print 'Is DK anti-symmetric with respect to ij?'
    error_max = 0
    for i in range(0,N):
        for j in range(0,N):
            for a in range(0,DIM):
                error = np.linalg.norm( DK[i,j,:,:,a] + DK[j,i,:,:,a] )
                error_max = np.maximum( error, error_max )
    print 'max for DK_ij + DK_ji = ' + str( error_max )
    return 1

def test_metric ( x ):
    h = 1e-8

    gsharp,Dgsharp,D2gsharp,g,Dg = gs(x)
    Gamma = Gamma_g(gsharp,Dgsharp,D2gsharp,g,Dg)
    d0Gamma = DGamma_g(gsharp,Dgsharp,D2gsharp,g,Dg)


    delta = np.identity(N*DIM)

    error_max = 0.
    x_a = np.copy(x)
    for i in range(0,N*DIM):
        x_a[i] = x[i] + h
        gsharp_a,Dgsharp_a,D2gsharp_a,g_a,Dg_a = gs(x_a)
        for l,j in itertools.product(range(N*DIM),range(N*DIM)):
            der = ( gsharp_a[l,j] - gsharp[l,j] ) / h
            error = np.linalg.norm(  der - Dgsharp[l,j,i] )
            error_max = np.maximum(error, error_max)
        x_a[i] = x[i]
    print 'error_max for Dgsharp = ' + str( error_max )
    if (error_max > 100*h):
        print 'WARNING:  COMPUTATION OF Dgsharp APPEARS TO BE INACCURATE'

    error_max = 0.
    x_a = np.copy(x)
    for (i,k) in itertools.product(range(0,N*DIM),range(N*DIM)):
        x_a[i] = x[i] + h
        gsharp_a,Dgsharp_a,D2gsharp_a,g_a,Dg_a = gs(x_a)
        for l,j in itertools.product(range(N*DIM),range(N*DIM)):
            der = ( Dgsharp_a[l,j,k] - Dgsharp[l,j,k] ) / h
            error = np.linalg.norm(  der - D2gsharp[l,j,k,i] )
            error_max = np.maximum(error, error_max)
        x_a[i] = x[i]
    print 'error_max for D2gsharp = ' + str( error_max )
    if (error_max > 100*h):
        print 'WARNING:  COMPUTATION OF D2gsharp APPEARS TO BE INACCURATE'

    error_max = 0.
    x_a = np.copy(x)
    for i in range(0,N*DIM):
        x_a[i] = x[i] + h
        gsharp_a,Dgsharp_a,D2gsharp_a,g_a,Dg_a = gs(x_a)
        for l,j in itertools.product(range(N*DIM),range(N*DIM)):
            der = ( g_a[l,j] - g[l,j] ) / h
            #print  "%g / %g " % (der , Dgsharp[l,j,i] )
            error = np.linalg.norm(  der - Dg[l,j,i] )
            error_max = np.maximum(error, error_max)
        x_a[i] = x[i]
    print 'error_max for Dg = ' + str( error_max )
    if (error_max > 100*h):
        print 'WARNING:  COMPUTATION OF Dg APPEARS TO BE INACCURATE'

    error_max = 0.
    x_a = np.copy(x)
    for i in range(0,N*DIM):
        x_a[i] = x[i] + h
        gsharp_a,Dgsharp_a,D2gsharp_a,g_a,Dg_a = gs(x_a)
        Gamma_a = Gamma_g(gsharp_a,Dgsharp_a,D2gsharp_a,g_a,Dg_a)
        d0Gamma_a = DGamma_g(gsharp_a,Dgsharp_a,D2gsharp_a,g_a,Dg_a)
        for k,l,j in itertools.product(range(N*DIM),range(N*DIM),range(N*DIM)):
            der = ( Gamma_a[k,l,j] - Gamma[k,l,j] ) / h
            error = np.linalg.norm(  der - d0Gamma[k,l,j,i] )
            error_max = np.maximum(error, error_max)
        x_a[i] = x[i]
    print 'error_max for d0Gamma = ' + str( error_max )
    if (error_max > 100*h):
        print 'WARNING:  COMPUTATION OF d0Gamma APPEARS TO BE INACCURATE'

    print 'TESTING SYMMETRIES'
    print 'Is g symmetric with respect to ij?'
    print 'max for g_ij - g_ji = ' + str( np.linalg.norm(g-g.T,np.inf) )
    print 'Is gsharp symmetric with respect to ij?'
    print 'max for g^ij - g^ji = ' + str( np.linalg.norm(gsharp-gsharp.T,np.inf) )
    print 'Are g and gsharp inverses?'
    print 'g^ikg_kj-delta = ' + str( np.linalg.norm(np.einsum('ik,kj->ij',gsharp,g)-delta,np.inf) )
    print 'g_ikg^kj-delta = ' + str( np.linalg.norm(np.einsum('ik,kj->ij',g,gsharp)-delta,np.inf) )
    print 'Is Gamma^k_ij symmetric with respect to ij?'
    error_max = 0
    for k,i,j in itertools.product(range(N*DIM),range(N*DIM),range(N*DIM)):
        error = np.linalg.norm( Gamma[k,i,j] - Gamma[k,j,i] )
        error_max = np.maximum( error, error_max )
    print 'max for Gamma^k_ij - Gamma^k_ji = ' + str( error_max )

    return 1

def test_functions( trials ):
    #checks that each function does what it is supposed to
    h = 10e-7
    x = SIGMA*np.random.randn(N*DIM)
    Xa = SIGMA*np.random.randn(N*DIM,rank)
    xi = SIGMA*np.random.randn(N*DIM)
    xia = np.random.randn(N*DIM,rank)
    
    test_kernel_functions( x.reshape([N,DIM]) )
    test_metric( x )

    s = weinstein_darboux_to_state( x , Xa, xi, xia )
    ds = ode_function( s , 0 )
    dx,dXa,dxi_coded,dxia_coded = state_to_weinstein_darboux( ds ) 

    print 'a test of the ode:'
    #print 'dxi_coded =' + str(dxi_coded)
    xh = np.copy(x)
    dxi_estim = np.zeros([N*DIM])
    for i in range(0,N*DIM):
        xh[i] = x[i] + h
        dxi_estim[i] = - ( Hamiltonian(xh,Xa,xi,xia) - Hamiltonian(x,Xa,xi,xia) ) / h 
        xh[i] = xh[i] - h
    #print 'dxi_estim =' + str(dxi_estim)
    #print 'dxi_coded =' + str(dxi_coded)
    #print 'dxi_error =' + str(dxi_estim - dxi_coded)
    print 'dxi_error =' + str(np.linalg.norm(dxi_estim - dxi_coded,np.inf))

    #print 'dxia_coded =' + str(dxia_coded)
    Xah = np.copy(Xa)
    dxia_estim = np.zeros([N*DIM,rank])
    for i in range(0,N*DIM):
        for a in range(rank):
            Xah[i,a] = Xa[i,a] + h
            dxia_estim[i,a] = - ( Hamiltonian(x,Xah,xi,xia) - Hamiltonian(x,Xa,xi,xia) ) / h 
            Xah[i,a] = Xah[i,a] - h
    #print 'dxia_estim =' + str(dxia_estim)
    #print 'dxia_coded =' + str(dxia_coded)
    #print 'dxia_error =' + str(dxia_estim - dxia_coded)
    print 'dxia_error =' + str(np.linalg.norm(dxia_estim - dxia_coded,np.inf))

    if trials >  1:
        test_functions(trials-1)


def integrate(state, N_t=20, T=1., pts=None):
    """
    flow forward integration

    Points pts are carried along the flow without affecting it
    """
    init()

    t_span = np.linspace(0. ,T , N_t )
    #print 'forward integration: SIGMA = ' + str(SIGMA) + ', N = ' + str(N) + ', DIM = ' + str(DIM) + ', N_t = ' + str(N_t) + ', T = ' + str(T)

    odef = ode_function
    
    if pts == None:
        y_span = odeint( odef , state , t_span)
        return (t_span, y_span)
    else:
        y_span = odeint( odef , np.hstack((state,pts.flatten())) , t_span)
        return (t_span, y_span[:,0:get_dim_state()], y_span[:,get_dim_state():])


def energy(state):
    x,Xa,xi,xia = state_to_weinstein_darboux( state )

    return Hamiltonian(x,Xa,xi,xia)

