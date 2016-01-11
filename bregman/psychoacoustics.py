#psychoacoustics.py - psychoacoustic parameters and processing
# Bregman - python toolkit for music information retrieval

__version__ = '1.0'
__author__ = 'Michael A. Casey'
__copyright__ = "Copyright (C) 2010  Michael Casey, Dartmouth College, All Rights Reserved"
__license__ = "GPL Version 2.0 or Higher"
__email__ = 'mcasey@dartmouth.edu'


import numpy as np

def dB_to_amp(dB):
    """
    ::

        Convert decibels to amplitude
    """
    return 10**(dB/20.)

def amp_to_dB(amp):
    """
    ::

        Convert amplitude to decibels
    """
    return 20*np.log10(amp)

def power_to_dB(pwr):    
    """
    ::

        Convert power to decibels
    """
    return 10*np.log10(pwr)

def hertz_to_bark(cf):
    """
    ::

        Convert frequency in Hz to Bark band 
    """
    return 13 * np.arctan(0.00076 * cf) + 3.5 * np.arctan( (cf / 7500.)**2)

def bark_to_critical_bandwidth(bark):
    """
    ::

        Return critical bandwidth for given Bark band
    """
    bw = 52548. / ( bark**2 - 52.56 * bark + 690.39 )
    return bw

def hertz_to_critical_bandwidth(cf):
    """
    ::

        Convert frequency in Hertz to critical bandwidth
    """
    bark = hertz_to_bark(cf)
    bw = bark_to_critical_bandwidth(bark)
    return bw

def dissonance_fun(freqs, amps=None, params=None):
    """
    ::

        Compute dissonance between partials with center frequencies in freqs,
        and amplitudes in amps. Based on William Sethares after Plomp and Levelt:

        default params = (-3.51, -5.75, 0.0207, 19.96, 5, -5, 0.24)
        default amps, use 1 as amplitude for all partials.
    """
    if params == None: params = (-3.51, -5.75, 0.0207, 19.96, 5, -5, 0.24)
    b1, b2, s1, s2, c1, c2, Dstar  = params
    if amps == None: amps = [1]*len(freqs)
    f = np.array(freqs)
    a = np.array(amps)
    idx = np.argsort(f)
    f = f[idx]
    a = a[idx]
    N = f.size
    D = 0
    for i in range(1, N):
        Fmin = f[ 0 : N - i ]
        S = Dstar / ( s1 * Fmin + s2)
        Fdif = f[ i : N ] - f[ 0 : N - i ]
        am = a[ i : N ] * a[ 0 : N - i ]
        Dnew = am * (c1 * np.exp (b1 * S * Fdif) + c2 * np.exp(b2 * S * Fdif))
        D += Dnew.sum()
    return D

