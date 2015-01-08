# MUSIC014/102 - Music, Information, Neuroscience, 
# Week 1 Lab
#   Using the Plompt and Levelt dissonance function
#
# Professor Michael Casey, 1/7/2015

from pylab import *
from bregman.suite import *
import scipy.signal as signal
import pdb

def ideal_chromatic_dissonance(num_harmonics=7, f0=440):
    """
    One octave of chromatic dissonance values
    """
    harms = arange(num_harmonics)+1
    freqs = [f0*i for i in harms]
    amps = [exp(-.5*i) for i in harms]
    freqs2 = array([[f0*2**(k/12.)*i for i in harms] for k in range(0,13)])

    all_amps = r_[amps,amps]
    diss = []
    for f in freqs2:
        all_freqs = r_[freqs,f]
        idx = all_freqs.argsort()
        diss.append(dissonance_fun(all_freqs[idx], all_amps[idx]))
    return array(diss)

def get_peaks(F):
    """
    Extract peaks from linear spectrum in F
    Algorithm 1: zero-crossings of derivative of smoothed spectrum
    """
    X = F.X.copy()
    b,a = signal.butter(10, .25) # lp filter coefficients
    # Smoothing
    signal.filtfilt(b,a,X,axis=0)
    # Derivative
    Xd = diff(X,axis=0)
    # Zero crossing
    thresh=1e-9
    peak_idx = []
    for i,x in enumerate(Xd.T):
        idx = where((x[:-1]>thresh)&(x[1:]<-thresh))[0] + 1
        if len(idx):
            idx = idx[X[idx,i].argsort()][::-1]
        peak_idx.append(idx)
    return peak_idx

def audio_chromatic_scale(f0=440, num_harmonics=7):
    N = 11025
    nH = num_harmonics
    H = vstack([harmonics(f0=f0*2**(k/12.),num_harmonics=nH, num_points=N) for k in arange(13)])
    return H

def audio_chromatic_dissonance(f0=440, num_harmonics=7, num_peaks=10):
    sr = 44100
    nfft = 8192
    afreq = sr/nfft
    H = audio_chromatic_scale(f0=f0, num_harmonics=num_harmonics)
    h0 = H[0]
    diss = []
    for i,h in enumerate(H):
        F = LinearFrequencySpectrum((h0+h)/2.,nfft=nfft,wfft=nfft/2,nhop=nfft/4)
        P = get_peaks(F)
        frame = []
        for j,p in enumerate(P):
            freqs = afreq*p[:num_peaks] # take middle frame as reference
            mags = F.X[p[:num_peaks],j]
            idx = freqs.argsort()
            frame.append(dissonance_fun(freqs[idx],mags[idx]))        
        diss.append(array(frame).mean())
    return array(diss)


def dissonance_plot(f0=440, num_harmonics=7, num_peaks=10):
    figure()
    diss_i = ideal_chromatic_dissonance(f0=f0, num_harmonics=num_harmonics)
    diss = audio_chromatic_dissonance(f0=f0, num_harmonics=num_harmonics, num_peaks=num_peaks)
    plot(diss_i / diss_i.max(), linestyle='--', linewidth=2)
    plot(diss / diss.max())
    t_str = 'f0=%d, partials=%d, peaks=%d'%(f0,num_harmonics,num_peaks)
    title('Dissonance (chromatic): '+t_str,fontsize=16)
    legend(['ideal','estimated'])
    xlabel('Pitch class (chroma)',fontsize=14)
    ylabel('Dissonance',fontsize=14)
    grid()
