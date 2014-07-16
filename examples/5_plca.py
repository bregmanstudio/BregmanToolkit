# examples_plca.py 
# Bregman sound-mixture modeling separation examples
#
# Copyright (C) 2011 Mike Casey
# Dartmouth College, Hanover, NH
# All Rights Reserved
#
# Usage:
#
# from examples_plca import *
# audio_file = os.path.join(audio_dir,'amen.wav')
# x, sr, fmt = wavread(audio_file) # load audio file
# p = Features.default_params()    # get default features
# p['nhop']=2048                   # set hop size to 2048
#
# #1. PLCA, run the following examples
# help ex_1a
# w,z,h = ex_1a(x, p, n=4)
#
# help ex_1b
# #try the following with different values of alphaZ, alphaH, n
# w,z,h = ex_1b(x, p, n=4, alphaZ=-0.01, alphaH=-0.00001)
#
# #2. Shift invariant PLCA (time-shift only)
# #Using a 2D basis with a 1D convolution kenel for each component
#
# help ex_2a
# # Try the following with different values of n and win
# w,z,h = ex_2a(x, p, n=4, win=5)
# imagesc(w[:,0,:]) # show first basis function
# imagesc(w[:,0,:]) # show second basis function
# # etc..
# figure()
# plot(h[0,:]) # plot first time kernel
# plot(h[1,:]) # plot second time kernel
#
# help ex_2b
# # Try the following with different values of alphaZ, alphaH, win
# w,z,h = ex_2b(x, p, n=10, win=5, alphaZ=-0.01, alphaH=-0.00001):
# 
# #3. 2D shift-invariant PLCA, no shift in frequency
# help ex_3a
# # Try the following with different values of n and win=(1,10) and (1,3)
# w,z,h = ex_3a(x, p, n=4, win=(1,5))
# imagesc(w[:,0,:]) # show first basis function
# imagesc(h[0,:,:]) # show first 2D convolution kernel (actually 1D, no freq shift)
#
# help ex_3b
# w,z,h = ex_3b(x, p, n=1, win=(24,5))
# imagesc(w[:,0,:]) # show only basis function
# imagesc(h[0,:,:]) # show only 2D convolution kernel (with freq shifts)
#
# # 4. Doing it on a pitched example
# audio_file = os.path.split(bregman.__file__)[0]+os.sep+'audio'+os.sep+'gmin.wav'
# x, sr, fmt = wavread(audio_file)
# p['nbpo'] = 36 # increase CQFT frequency resolution
# w,z,h = ex_3b(x, p, n=1, win=(36,5))
# imagesc(w[:,0,:]) # show only basis function
# imagesc(h[0,:,:]) # show only 2D convolution kernel (with freq shifts)
# # The latter is a piono roll transcription, with wrapping at the extremes of the pitch range 
# # So, we have essentially achived an automatic transcription.
#
# # 5.
# # For a 20s-30s piece of music of your choosing, perform a 2D shift-invariant analysis
# # optimize the parameters so that your analysis "makes sense"
# # Be prepared to present your analysis, and any interesting conclusions, in class
# # Ideas: finding riffs in pop music, analysis of sound objects in Musique Concrete, analysis of texture in Ligeti or Lachemann
# # 

from bregman.suite import *
from pylab import *
import os
import os.path

# Global feature settings
p = Features.default_params()
p['feature']='cqft'
p['nhop']=2048

def play_components(X_hat, F, x=None):
    # play individual separated feature spectrograms as audio
    # separated spectrograms are stored in a list
    for Xh in X_hat:
        x_hat = F.inverse(Xh) # invert to audio to F.x_hat
        x = x_hat if x is None else x
        x = x.mean(1) if len(x.shape)>1 else x
        xn = min(len(x), len(x_hat))
        x_orig = 0.1*atleast_1d(x[:xn]).T
        x_hat = atleast_1d(x_hat[:xn] / (x_hat[:xn].max() + 0.01))
        play(c_[x_orig, x_hat].T, F.feature_params['sample_rate'])

def invert_component(cls, w, z, h):
    w = atleast_2d(w)
    if cls==PLCA: w = w.T 
    h = atleast_2d(h)
    return cls.reconstruct(w,z,h)

def ex_1a(x, p, n=4):
    # ex_1a(x, n=4): PLCA into n=4 components         
    # 'p' dict contains feature parameters
    print "\nExample 1: Standard PLCA with %s"%p['feature'],"features split into n=%d"%n,"components"
    F = Features(x, p)
    w,z,h,norm,recon,logprob = PLCA.analyze(F.X, n)
    X_hat = [invert_component(PLCA, w[:,k], z[k], h[k,:]) for k in range(len(z))]
    play_components(X_hat, F, x)
    return w,z,h

def ex_1b(x, p, n=10, alphaZ=-0.01, alphaH=-0.00001):
    # ex_1b(x, n=10): PLCA with Dirichlet priors alphaZ,alphaH for sparsity on Z and H respectively.
    # start with 10 components, sparseness prior on Z will eliminate low probability components
    # 'p' dict contains feature parameters
    #  Dirichlet priors are shown as alphaZ and alphaH parameters:
    #    -ve values mean more sparse, +ve less sparse, 0 means an uninformative prior (the default).
    # If alphaZ < 0 (e.g. -0.01) the number of components will be truncated
    # If alphaW < 0 (e.g. -0.00001) the frequency functions will be sparse
    # If alphaH < 0 (e.g. -0.00001) the time functions will be sparse
    print "\nExample 1: Standard PLCA with %s"%p['feature'],"features split into n=%d"%n,"components using Dirichlet priors (alphaZ=%1.6f, alphaH=%1.6f) on component distribution sparseness."%(alphaZ, alphaH)    
    F = Features(x, p)
    w,z,h,norm,recon,logprob = PLCA.analyze(F.X, n, alphaZ=alphaZ, alphaH=alphaH)
    X_hat = [invert_component(PLCA, w[:,k], z[k], h[k,:]) for k in range(len(z))]
    play_components(X_hat, F, x)
    return w,z,h
    
def ex_2a(x, p, n=4, win=5):
    # ex_2a(x, n=4, win=5): shift-invariant SIPLCA into 4 components
    # produces a two-dimensional basis that is shift-invariant in time only.
    # 'p' dict contains feature parameters
    # We can use either STFT or CQFT because we only require a time shiftable basis
    # Win is the duration, in frames, of the basis function.
    # 
    print "\nExample 2: Time-shift invariant PLCA with %s"%p['feature'],"features split into n=%d"%n,"components, component window length=%d"%win
    F = Features(x, p)
    w,z,h,norm,recon,logprob = SIPLCA.analyze(F.X, n, win=win)
    X_hat = [invert_component(SIPLCA, w[:,:,k], z[k], h[k,:]) for k in range(len(z))]
    play_components(X_hat, F, x)
    return w,z,h

def ex_2b(x, p, n=10, win=5, alphaZ=-0.01, alphaH=-0.00001):
    # ex_2b(x, n=10, win=5): SIPLCA with Dirichlet prior for sparsity on Z and H, start with 10 components
    # produces a two-dimensional basis that is shift-invariant in time only.
    # 'p' dict contains feature parameters
    # We can use either STFT or CQFT because we only require a time shiftable basis
    print "\nExample 2: Time-shift invariant PLCA with %s"%p['feature'],"features split into n=%d"%n,"components, component window length=%d"%win, "using Dirichlet priors (alphaZ=%1.6f, alphaH=%1.6f) on component distribution sparseness."%(alphaZ, alphaH)    
    F = Features(x, p)
    w,z,h,norm,recon,logprob = SIPLCA.analyze(F.X, n, win=win, alphaZ=alphaZ, alphaH=alphaH)
    X_hat = [invert_component(SIPLCA, w[:,:,k], z[k], h[k,:]) for k in range(len(z))]
    play_components(X_hat, F, x)
    return w,z,h

def ex_3a(x, p, n=4, win=(1,5)):
    # ex_3a(x, n=4, win=(1,5)): SIPLCA2 into n=4 compnents using time and frequency shift invariance (two dimensional invariance)
    # Shift invariance in frequency requires a shiftable representation.
    # 'p' dict contains feature parameters
    # We use CQFT which has a logarithmic frequency axis. Transpositions are shifts in frequency.
    # The window fundtion is a tuple, with win[0] the number of shifts, and win[1] the duration of the basis function.
    print "\nExample 3: Time-frequency-shift invariant PLCA with %s"%p['feature'],"features split into n=%d"%n,"components, component num shifts=%d,window length=%d"%win    
    F = Features(x, p)
    w,z,h,norm,recon,logprob = SIPLCA2.analyze(F.X, n, win=win)
    X_hat = [invert_component(SIPLCA2, w[:,k,:], z[k], h[k,:,:]) for k in range(len(z))]
    play_components(X_hat, F, x)
    return w,z,h    

def ex_3b(x, p, n=1, win=(24,5)):
    # ex_3b(x, n=1, win=(24,5)): SIPLCA2 into 1 pitch=-shifted sparse component
    # 2D Shift invariance in frequency requires a shiftable representation.
    # 'p' dict contains feature parameters
    # Use CQFT, which has a logarithmic frequency axis, so that transpositions are shifts.
    # 
    print "\nExample 3: Time-frequency-shift invariant PLCA with %s"%p['feature'],"features split into n=%d"%n,"components, component num shifts=%d,window length=%d"%win    
    F = Features(x, p)
    w,z,h,norm,recon,logprob = SIPLCA2.analyze(F.X, n, win=win)
    X_hat = [invert_component(SIPLCA2, w[:,k,:], z[k], h[k,:,:]) for k in range(len(z))]
    play_components(X_hat, F, x)
    return w,z,h    

def separate(clf, x, p, n=5, **kwargs):
    # meta function to call PLCA class clf, on audio x, num components n
    # 'p' dict contains feature parameters
    # **kwargs: win=(shifts,len) for frequency and time shift invariance, 
    # alphaW Dirichlet pior on W basis, alphaZ Dircihlet prior on Z, alphaH Dirichlet prior on H
    #
    F = Features(x, p)
    w,z,h,norm,recon,logprob = clf.analyze(F.X, n, **kwargs)
    if clf==SIPLCA2:
        X_hat = [invert_component(clf, w[:,k,:], z[k], h[k,:,:]) for k in range(len(z))]        
    else:
       X_hat = [invert_component(clf, w[:,:,k], z[k], h[k,:]) for k in range(len(z))]
    play_components(X_hat, F, x)    
    return w,z,h,X_hat,F


if __name__ == "__main__":
    audio_file = os.path.join(audio_dir,'amen.wav')
    x, sr, fmt = wavread(audio_file)

    p['feature']='cqft'
    p['nfft']=16384
    p['wfft']=8192
    p['nhop']=2048

    ex_1a(x, p, 4)

    
