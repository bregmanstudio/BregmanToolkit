# examples_features.py 
# Bregman audio feature analysis
#
# Copyright (C) 2011 Mike Casey
# Dartmouth College, Hanover, NH
# All Rights Reserved
#

from bregman.suite import *
import os
import os.path
from pylab import *

# Examples of using the Features base class to extract features
# and the derived helper classes to extract the same features.

def ex_1a(x):
    """
    Example 1a: magnitude STFT analysis of audio file
    Window parameters set for narrow band viewing (harmonics)
    nfft = 16384, wfft = 8192, nhop = 2205 (20Hz frame rate)
    """
    # LinearFrequencySpectrum helper class
    F = LinearFrequencySpectrum(x, nfft=16384, wfft=8192, nhop=2205)
    F.feature_plot(dbscale=True, normalize=True)    

    ## Alternate extraction method
    ## Same feature extractor using base class and parameter dict
    # p = Features.default_params()
    # p['feature'] = 'stft'
    # p['nfft'] = 16384
    # p['wfft'] = 8192
    # p['nhop'] = 2205 
    # F = Features(x,p)

    ## Alternate extraction method 2
    ## Same feature extractor using base class and alternate parameter dict method    
    # F = Features(x, {'feature':'stft', 'nfft':16384, 'wfft':8192, 'nhop':2205} )
    return F

def ex_1b(x):
    """
    Example 1b: magnitude STFT analysis of audio file
    Window parameters set for wide band viewing (formants)
    nfft=256, wfft=256, nhop=256
    """

    F = LinearFrequencySpectrum(x, nfft=256, wfft=256, nhop=256)
    F.feature_plot(dbscale=True, normalize=True)
    title('Wide-band magnitude short-time Fourier transform (STFT)')

    ## Alternate extraction method
    # Same feature extractor using base class and parameter dict
    # p = Features.default_params()
    # p['feature'] = 'stft'
    # p['nfft'] = 256
    # p['wfft'] = 256
    # p['nhop'] = 256 
    # F = Features(x,p)

    ## Alternate extraction method 2
    ## Same feature extractor using base class and alternate parameter dict method    
    # F = Features(x, {'feature':'stft', 'nfft':256, 'wfft':256, 'nhop':256} )
    return F

def ex_1c(x):
    """
    Example 1c: magnitude STFT analysis of audio file
    Window parameters set for trade-off between narrow and wide band analysis
    (harmonics + formants)
    """
    F = LinearFrequencySpectrum(x, nfft=1024, wfft=512, nhop=512)
    F.feature_plot(dbscale=True, normalize=True)
    title('Medium-band magnitude short-time Fourier transform (STFT)')

    ## Alternate extraction method
    # p = Features.default_params()
    # p['feature'] = 'stft'
    # p['nfft'] = 1024
    # p['wfft'] = 512
    # p['nhop'] = 512 
    # F = Features(x,p)

    ## Alternate extraction method 2
    # F = Features(x, {'feature':'stft', 'nfft':1024, 'wfft':512, 'nhop':512} )
    return F

def ex_2a(x):
    """
    Example 2a: constant-Q magnitude analysis of audio file    
    Constant-Q frequency resolution is 12 bands per octave
    nfft = 16384, wfft = 8192, nhop = 2205 (20Hz frame rate)    
    """
    F = LogFrequencySpectrum(x, nfft=16384, wfft=8192, nhop=2205)
    F.feature_plot(dbscale=True, normalize=True)
    title('12 Bands-per-Octave Constant-Q Fourier Transform (CQFT)')

    ## Alternate extraction method
    # p = Features.default_params()
    # p['feature'] = 'cqft'
    # p['nfft'] = 16384
    # p['wfft'] = 8192
    # p['nhop'] = 2205 
    # p['nbpo'] = 12
    # F = Features(x,p)

    ## Alternate extraction method 2
    # F = Features(x, {'feature':'stft', 'nfft':16384, 'wfft':8192, 'nhop':2205, 'nbpo':12})
    return F 

def ex_3a(x):
    """
    Example 3a: 12-band chromagram analysis of audio file    
    Constant-Q frequency resolution is 12 bands per octave
    nfft = 16384, wfft = 8192, nhop = 2205 (20Hz frame rate)    
    """
    F = Chromagram(x, nfft=16384, wfft=8192, nhop=2205)
    F.feature_plot(dbscale=True, normalize=True)
    title('12-Band Chromagram')

    ## Alternate extraction method
    # p = Features.default_params()
    # p['feature'] = 'chroma'
    # p['nfft'] = 16384
    # p['wfft'] = 8192
    # p['nhop'] = 2205 
    # p['nbpo'] = 12
    # F = Features(x,p)

    ## Alternate 2
    # F = Features(x, {'feature':'chroma', 'nfft':16384, 'wfft':8192, 'nhop':2205, 'nbpo':12})
    return F 

def ex_4a(x):
    """
    Example 4a, MFCCs: 10-band Mel-Frequency Cepstral Coefficients
    nfft = 16384, wfft = 8192, nhop = 2205, ncoef=10    
    """
    F = LogFrequencyCepstrum(x, nhop=2205)
    F.feature_plot(normalize=True)
    title('Mel-Frequency Cepstral Coefficients')
    return F 

def ex_4b(x):
    """
    Example 4b: Liftered 10 Mel-Frequency Cepstral Coefficients inverted back to Log Spectrum
    nfft = 16384, wfft = 8192, nhop = 2205, ncoef=10    
    """
    F = LowQuefrencyLogFrequencySpectrum(x, nhop=2205)
    F.feature_plot(dbscale=True, normalize=True)
    title('Low-Quefrency Log-Frequency Spectrum')
    return F 


if __name__ == "__main__":
    audio_file = os.path.split(bregman.__file__)[0]+os.sep+'audio'+os.sep+'gmin.wav'

    F1a = ex_1a(audio_file)
    F1b = ex_1b(audio_file)
    F1c = ex_1c(audio_file)
    F2a = ex_2a(audio_file)
    F3a = ex_3a(audio_file)
    F4a = ex_4a(audio_file)
    F4b = ex_4b(audio_file)

    
