#!/usr/bin/python

import matplotlib.pyplot as plt
import mpp
import numpy as np

R = np.load('output/setup.npy')
N = mpp.N = int(R[0])
DIM = mpp.DIM = int(R[1])
SIGMA = mpp.SIGMA = int(R[2])
mpp.init()

print "setup: N %d, DIM %d, SIGMA %g" % (mpp.N,mpp.DIM,mpp.SIGMA)

DIM = 2

def display_velocity_field( x , Xa, xi, xia , q1=None ):
    Window = 2
    res = 40
    N_nodes = res**DIM
    store = np.outer( np.linspace(-Window,Window , res), np.ones(res) )
    nodes = np.zeros( [N_nodes , mpp.DIM] )
    nodes[:,0] = np.reshape( store , N_nodes )
    nodes[:,1] = np.reshape( store.T , N_nodes )
    q = x.reshape([-1,DIM])
    K,DK,_ = mpp.derivatives_of_kernel( nodes , q )
    # compute g momentum
    gsharp,Dgsharp,D2gsharp,g,Dg = mpp.gs(x)
    Gamma = mpp.Gamma_g(gsharp,Dgsharp,D2gsharp,g,Dg)
    GammaX = np.einsum('hja,ag->hgj',Gamma,Xa)
    delta = np.eye(N*DIM)
    W = np.einsum('ab,ka,lb->kl',delta,Xa,Xa)        
    dx = np.einsum('ij,j->i',W,xi)-np.einsum('ih,jbh,jb->i',W,GammaX,xia)
    p = np.einsum('ij,j->i',g,dx).reshape([N,DIM])
    vel_field = np.einsum('ijab,jb->ia',K,p)
    U = vel_field[:,0]
    V = vel_field[:,1]
    f = plt.figure(1)
    plt.quiver( nodes[:,0] , nodes[:,1] , U , V , color='0.50' )
    plt.plot(q[:,0],q[:,1],'ro')

    # generate vertices of a circle
    N_vert = 20
    circle_verts = np.zeros( [ 2 , N_vert + 1 ] )
    theta = np.linspace(0,2*np.pi, N_vert )
    circle_verts[0,0:N_vert] = 0.2*np.cos(theta)
    circle_verts[1,0:N_vert] = 0.2*np.sin(theta)
    verts = np.zeros([2, N_vert + 1])
    units = np.ones( N_vert + 1)

    for i in range(0,len(q)):
        plt.arrow(q[i,0], q[i,1], 0.2*p[i,0], 0.2*p[i,1],\
            head_width=0.2, head_length=0.2,\
            fc='b', ec='b')
        if (q1 != None):
            verts = np.dot(q1[i,:,:], circle_verts ) \
                + np.outer(q[i,:],units)
            print np.shape( verts )
            print np.shape( q1 )
            plt.plot(verts[0],verts[1],'b-')

        plt.axis([- Window, Window,- Window, Window ])
        plt.axis('equal')
    return f

y_data = np.load('output/state_data.npy')
time_data = np.load('output/time_data.npy')

#print 'shape of y_data is ' + str( y_data.shape )
I = range(y_data.shape[0])
N_timestep = len(I)
print 'generating png files'
for k in I:
    #print "step %d of %d, t %g" % (k,N_timestep,time_data[k])
    x,Xa,xi,xia = mpp.state_to_weinstein_darboux( y_data[k] )
    f = display_velocity_field(x,Xa,xi,xia)
    time_s = str(time_data[k])
    plt.suptitle('t = '+ time_s[0:4] , fontsize=16 , x = 0.75 , y = 0.25 )
    fname = './movie_frames/frame_'+str(k)+'.png'
    f.savefig( fname, dpi=200 )
    plt.close(f)
print 'done'
