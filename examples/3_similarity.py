# examples_similarity.py 
# Bregman distance function examples
#
# Copyright (C) 2011 Mike Casey
# Dartmouth College, Hanover, NH
# All Rights Reserved
#

from bregman.suite import *
import os
import os.path
from pylab import *


def ex_1a(x, ttl=''):
    """
    Self-similarity matrix of signal x, using CQFT featurs
    """
    F = LogFrequencySpectrum(x, nhop=2205) 
    D = distance.euc_normed(F.X.T,F.X.T)
    imagesc(D)
    title(ttl)
    return D

def ex_1b(x, ttl=''):
    """
    Self-similarity matrix of signal x, using overlapping stacked CQFT features (shingles)
    """    
    F = LogFrequencySpectrum(x, nhop=2205) 
    X = adb.stack_vectors(F.X.T, 5, 1)
    D = distance.euc_normed(X,X)
    imagesc(D)
    title(ttl)
    return D

def ex_2a(x, ttl=''):
    """
    Self-similarity matrix of signal x, using CHROMA features
    """
    F = Chromagram(x, nhop=2205) 
    D = distance.euc_normed(F.X.T,F.X.T)
    imagesc(D)
    title(ttl)
    return D

def ex_2b(x, ttl=''):
    """
    Self-similarity matrix of signal x, using overlapping stacked CHROMA features (shingles)
    """    
    F = Chromagram(x, nhop=2205) 
    X = adb.stack_vectors(F.X.T, 5, 1)
    D = distance.euc_normed(X,X)
    imagesc(D)
    title(ttl)
    return D

def ex_3a(x, y, ttl=''):
    """
    Cross similarity matrix of two signals, using CHROMA features
    """
    F = Chromagram(x, nhop=2205) 
    G = Chromagram(y, nhop=2205) 
    D = distance.euc_normed(F.X.T,G.X.T)
    imagesc(D.T)
    title(ttl)
    xlabel('x vectors')
    ylabel('y vectors')    
    return D

def ex_4a(x, y, ttl=''):
    """
    Cross similarity matrix of two signals, using CHROMA features
    """
    F = Chromagram(x, nhop=2205) 
    G = Chromagram(y, nhop=2205) 
    D = distance.euc_normed(F.X.T,G.X.T)
    imagesc(D.T)
    title(ttl)
    xlabel('x vectors')
    ylabel('y vectors')    
    print("Original sound")
    play(x)
    print("Re-sampled sound")
    play(y)
    # find warp path to align signals
    p,q,W = dtw(D.T)
    plot(q,p,'w',linewidth=5)
    axis('tight')
    warp_idx = array([p[where(q>=i)[0][0]] for i in range(F.X.shape[1])])
    G = LinearFrequencySpectrum(y, nhop=2205)  # get invertible features
    x_hat = G.inverse(G.X[:,warp_idx],pvoc=1)
    print("Aligned sound: Features.inverse(F.X[:,warp_idx])")
    play(balance_signal(x_hat))
    return D

if __name__ == "__main__":
    audio_file_1 = os.path.split(bregman.__file__)[0]+os.sep+'audio'+os.sep+'gmin.wav'
    x, sr_x, fmt_x = wavread(audio_file_1)
    
    ex_1a(x, 'Self Similarity Matrix: CQFT')
    ex_1b(x, 'Self Similarity Matrix: stacked CQFT')

    ex_2a(x, 'Self Similarity Matrix: CHROMA')
    ex_2b(x, 'Self Similarity Matrix: stacked CHROMA')

    # Cross similarity examples require a second audio file. 
    # Warp x to create a simulated alternate performance y
    y = concatenate([resample(x[:len(x)/4], len(x)/4/2),
                    resample(x[len(x)/4:len(x)/2], len(x)/4),
                    resample(x[len(x)/2:3*len(x)/4], 2*len(x)/4),
                     x[3*len(x)/4:]])
    
    ex_3a(x,y, 'Cross Similarity Matrix: CHROMA')

    ex_4a(x,y, 'Dynamic Time Warp Path: CHROMA')
