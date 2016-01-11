# ditance.py - distance functions for Bregman toolbox
__version__ = '1.0'
__author__ = 'Michael A. Casey'
__copyright__ = "Copyright (C) 2010-2011  Michael Casey, Dartmouth College, All Rights Reserved"
__license__ = "GPL Version 2.0 or Higher"
__email__ = 'mcasey@dartmouth.edu'

import numpy as N
import pdb

def euc(A,B, old_algorithm=False):
    """ 
    ::

        d = euc(A,B)
        Return the Euclidean distance between two matrices.
        Second dimension (num columns) of A and B must be the same.
    """
    if old_algorithm:
        D = N.zeros((A.shape[0],B.shape[0]))
        for i in range(A.shape[0]):
            D[i,:] = ((A[i,:] - B)**2).sum(1)
        # Correct floating-point precision around 0
        D[N.where(D<N.finfo(N.float32).eps)]=0
        return N.sqrt(D)
    else:
        return euc2(A,B)

def euc2(A,B):
    """ 
    ::

        d = euc2(A,B) , a faster implementation of Euclidean distance
        Return the Euclidean distance between two matrices.
        Second dimension (num columns) of A and B must be the same.
    """
    r = A.shape[0]
    Anm = N.atleast_2d((A**2).sum(1)).repeat(r,0).T
    Bnm = N.atleast_2d((B**2).sum(1)).repeat(r,0)
    D = Anm+Bnm - 2 * N.dot(A,B.T)
    D[N.where(D<N.finfo(N.float32).eps)]=0
    return N.sqrt(D)

def euc_normed(A,B):
    """
    ::
    
        d = euc_normed(A,B)
        Return the normed Euclidean distance between two matrices
        Second dimension (num columns) of A and B must be the same
    """
    nz_idx = (A*A).sum(1) > 0
    AA = N.zeros(A.shape)
    BB = N.zeros(B.shape)    
    AA[nz_idx] = (A[nz_idx].T / N.sqrt((A[nz_idx]*A[nz_idx]).sum(1))).T
    nz_idx = (B*B).sum(1) > 0    
    BB[nz_idx] = (B[nz_idx].T / N.sqrt((B[nz_idx]*B[nz_idx]).sum(1))).T
    D = 2 - 2 * N.dot(AA,BB.T)
    D[N.where(D<N.finfo(N.float32).eps)]=0
    return N.sqrt(D)

def dot_normed(A,B):
    """
    ::
    
        d = dot_normed(A,B)
        Return the normed dot-product distance between two matrices
        Second dimension (num columns) of A and B must be the same
    """
    A = A.copy()
    B = B.copy()    
    A = (A.T / N.sqrt((A*A).sum(1))).T
    B = (B.T / N.sqrt((B*B).sum(1))).T
    D = N.dot(A,B.T)
    D[N.where(D<N.finfo(N.float32).eps)]=0
    return D

def cosine(A,B):
    """
    ::
        d = cosine(A,B)
        Return the cosine distance between two matrices
        Second dimension (num columns) of A and B must be the same
    """
    return 1 - dot_normed(A,B)
    
def corr_coeff(A,B):
    """
    ::
        d = corr_coeff(A,B):
        Return Pearson's product moment correlation coefficients
        Second dimension (num columns) of A and B must be the same        
    """
    A = A.copy()
    B = B.copy()
    A = A.T
    B = B.T
    A -= A.mean(0)
    std = A.std(0)
    std_nz = std != 0
    A[:, std_nz] /= std[std_nz] # avoid invariant features
    B -= B.mean(0)
    std = B.std(0)
    std_nz = std != 0
    B[:, std_nz] /= std[std_nz] # avoid invariant features
    D = N.dot(A.T,B) / (A.shape[0])
    return D

def corr_dist(A,B):
    """
    ::
        d = corr_dist(A,B):
        Return Pearson's product moment correlation distance
        Second dimension (num columns) of A and B must be the same        
    """
    D = 1.0 - corr_coeff(A,B)
    D[N.where(D<N.finfo(N.float32).eps)]=0        
    return D

def dtw(M):
    """ 
    ::

        p,q,D = dtw(M) 
        Use dynamic programming to find a min-cost path through matrix M.
        Return state sequence in p,q, and cost matrix in D     
        Ported from Matlab version by Dan Ellis, GPL v2
    """
    r,c = M.shape
    # costs
    D = N.zeros((r+1, c+1));
    D[0,:] = N.inf
    D[:,0] = N.inf
    D[0,0] = 0
    D[1:(r+1), 1:(c+1)] = M

    # traceback
    phi = N.zeros((r,c))
    for i in range(r):
        for j in range(c):
            dmax = N.min([D[i, j], D[i, j+1], D[i+1, j]])
            tb = N.argmin([D[i, j], D[i, j+1], D[i+1, j]])
            D[i+1,j+1] = D[i+1,j+1]+dmax
            phi[i,j] = tb

    # Traceback from top left
    i = r - 1 
    j = c - 1
    p = [i]
    q = [j]
    while i and j:
        tb = phi[i,j]
        if tb == 0:
            i = i - 1
            j = j - 1
        elif tb == 1:
            i = i - 1
        elif tb == 2:
            j = j - 1
        else:
            raise ValueError()
        p.insert(0,i)
        q.insert(0,j)

    # Strip off the edges of the D matrix before returning
    D = D[1:(r+1),1:(c+1)]
    return N.array(p),N.array(q),D

def mds(D, n=0, tol=0.9):  
    """
    ::

        P = mds(D,[n, tol])
        Multidimensional scaling of distance matrix D.

        inputs:
         D   - a distance matrix (similarity=0, dissimilarity=1)
              must be symmetric, positive, semidefinite,
              i.e. all values >=0 and D[i,j] == D[j,i]
         n   - dimensionality of result [default (automatic)]
         tol - tolerance [0-1] of auto selected dimensions [0.9] 

         outputs:
         P - a matrix of points in n dimensions
         s - Kruskal stress of solution
    """
    tol=0.99
    m = D.shape[0]    
    if n>m: n=m # dimensions cannot exceed points     
    # perform the MDS decomposition 
    A = -0.5 * D * D
    H = N.lib.twodim_base.eye(m) - 1./m; # idempotent H*H=H
    B = N.dot(N.dot(H, A), H)
#    print B
    # this should be a diagonal decomposition because B is symmetric 
    S, U = N.linalg.linalg.eig(B)
    idx = N.argsort(S)[::-1]
    p = S[idx].cumsum() / S.sum()
    if n==0: n = N.where(p>=tol)[0][0] + 1
    return S[idx[:n]]*U[:,idx[:n]], 1 - p[n-1]

def bhatt(d, z):
    """
    ::

        d = bhatt(d,z) 
        Bhattacharyya-weighted distance for vector of distances, d, and pair-wise prior products, z.
    """
    return 1.0 - z.sum() * N.exp(z - d).sum() / N.exp(z).sum()


def kl(a,b,symmetric=False):
    """
    ::
        d = kl(a,b, symmetric)
        Kullback-Leibler divergence for two PMFs/histograms
        a and b - vectors with the same dimensionality with sum(a)==sum(b)==1
        symmetric - Boolean flag to indicate use of symmetric kl
    """
    d = a * N.log( a / b )
    d[N.isnan(d)]=0
    d = d.sum()
    if symmetric:
        d = ( d + kl(b,a) ) / 2.0
    return d
    
    
    
