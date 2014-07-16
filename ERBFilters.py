# ERBFilters.py
# Python implementation of Slaney's ERB filter stuff, for getting dictionaries of kernel responses
#
# Created by Colin Raffel on 10/30/12

import numpy as np
import scipy.signal

# Computes an array of N frequencies uniformly spaced on an ERB scale
# Based on Slaney's Auditory Toolbox implementation
# Verified same output for input params 100, 44100/4, 100.
def ERBSpace( lowFreq, highFreq, N ):
  # Change the following three parameters if you wish to use a different
  # ERB scale.  Must change in MakeERBCoeffs too.
  # Glasberg and Moore Parameters
  EarQ = 9.26449
  minBW = 24.7
  order = 1
  
  # All of the followFreqing expressions are derived in Apple TR #35, "An
  # Efficient Implementation of the Patterson-Holdsworth Cochlear
  # Filter Bank."  See pages 33-34.
  return -(EarQ*minBW) + np.exp(np.arange(1, N+1)*(-np.log(highFreq + EarQ*minBW) + np.log(lowFreq + EarQ*minBW))/(1.0*N))*(highFreq + EarQ*minBW);

# Computes the filter coefficients for gammatone filters.
# Based on Slaney's Auditory Toolbox implementation.
# Verified same output for params 44100, 16, 100.
def makeERBFilters( fs, numChannels, lowFreq, highFreq=None ):
  
  if highFreq == None:
    highFreq = fs/2
  
  T = 1.0/fs
  cf = ERBSpace(lowFreq, highFreq, numChannels)
  
  # Change the followFreqing three parameters if you wish to use a different
  # ERB scale.  Must change in ERBSpace too.
  # Glasberg and Moore Parameters
  EarQ = 9.26449
  minBW = 24.7
  order = 1
  
  ERB = ((cf/EarQ)**order + minBW**order)**(1.0/order)
  B = 1.019*2*np.pi*ERB
  
  A0 = T
  A2 = 0
  B0 = 1
  B1 = -2*np.cos(2*cf*np.pi*T)/np.exp(B*T)
  B2 = np.exp(-2*B*T)
  
  A11 = -(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T) + 2*np.sqrt(3+2**1.5)*T*np.sin(2*cf*np.pi*T)/ \
          np.exp(B*T))/2.0
  A12 = -(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T) - 2*np.sqrt(3+2**1.5)*T*np.sin(2*cf*np.pi*T)/ \
          np.exp(B*T))/2.0
  A13 = -(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T) + 2*np.sqrt(3-2**1.5)*T*np.sin(2*cf*np.pi*T)/ \
          np.exp(B*T))/2.0
  A14 = -(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T) - 2*np.sqrt(3-2**1.5)*T*np.sin(2*cf*np.pi*T)/ \
          np.exp(B*T))/2.0
  
  gain = abs((-2*np.exp(4*1j*cf*np.pi*T)*T + \
              2*np.exp(-(B*T) + 2*1j*cf*np.pi*T)*T* \
              (np.cos(2*cf*np.pi*T) - np.sqrt(3 - 2**(3/2.0))* \
               np.sin(2*cf*np.pi*T))) * \
             (-2*np.exp(4*1j*cf*np.pi*T)*T + \
              2*np.exp(-(B*T) + 2*1j*cf*np.pi*T)*T* \
              (np.cos(2*cf*np.pi*T) + np.sqrt(3 - 2**(3/2.0)) * \
               np.sin(2*cf*np.pi*T)))* \
             (-2*np.exp(4*1j*cf*np.pi*T)*T + \
              2*np.exp(-(B*T) + 2*1j*cf*np.pi*T)*T* \
              (np.cos(2*cf*np.pi*T) - \
               np.sqrt(3 + 2**(3/2.0))*np.sin(2*cf*np.pi*T))) * \
             (-2*np.exp(4*1j*cf*np.pi*T)*T + 2*np.exp(-(B*T) + 2*1j*cf*np.pi*T)*T* \
              (np.cos(2*cf*np.pi*T) + np.sqrt(3 + 2**(3/2.0))*np.sin(2*cf*np.pi*T))) / \
             (-2 / np.exp(2*B*T) - 2*np.exp(4*1j*cf*np.pi*T) +  \
              2*(1 + np.exp(4*1j*cf*np.pi*T))/np.exp(B*T))**4)
  
  allfilts = np.ones( cf.shape[0] )
  return np.dstack( (A0*allfilts, A11, A12, A13, A14, A2*allfilts, B0*allfilts, B1, B2, gain) )[0]

# Process an input waveform with a gammatone filter bank
# Based on Slaney's Auditory Toolbox implementation
def ERBFilterBank(x, fcoefs ):
  A0  = fcoefs[:,0]
  A11 = fcoefs[:,1]
  A12 = fcoefs[:,2]
  A13 = fcoefs[:,3]
  A14 = fcoefs[:,4]
  A2  = fcoefs[:,5]
  B0  = fcoefs[:,6]
  B1  = fcoefs[:,7]
  B2  = fcoefs[:,8]
  gain= fcoefs[:,9]
  
  output = np.zeros( (gain.shape[0], x.shape[0]) )
  for chan in np.arange( gain.shape[0] ):
    y1 = scipy.signal.lfilter(np.array([A0[chan]/gain[chan], A11[chan]/gain[chan], \
                                        A2[chan]/gain[chan]]), \
                              np.array([B0[chan], B1[chan], B2[chan]]), x);
    y2 = scipy.signal.lfilter(np.array([A0[chan], A12[chan], A2[chan]]), \
                              np.array([B0[chan], B1[chan], B2[chan]]), y1);
    y3 = scipy.signal.lfilter(np.array([A0[chan], A13[chan], A2[chan]]), \
                              np.array([B0[chan], B1[chan], B2[chan]]), y2);
    y4 = scipy.signal.lfilter(np.array([A0[chan], A14[chan], A2[chan]]), \
                              np.array([B0[chan], B1[chan], B2[chan]]), y3);
    output[chan, :] = y4
  
  return output

# Converts a matrix of ERB filters from makeERBFilters to kernels using an input signal (eg impulse)
# It would be cool to make this non-ERB dependent, to try other filters.
def ERBFiltersToKernels( input, fcoefs, threshold = .001 ):
  # Get impulse responses
  impulseResponses = ERBFilterBank( input, fcoefs )
  # Dictionary for gammatone kernels
  kernelDictionary = {}
  for n in np.arange( impulseResponses.shape[0] ):
    impulseResponse = impulseResponses[n]
    impulseResponsePeak = np.max( impulseResponse )
    # Find index of last value greater than the threshold
    trim = 0
    for m in np.arange( impulseResponse.shape[0] ):
      if impulseResponse[m] > threshold*impulseResponsePeak:
        trim = m
    # Trim the impulse response to this value and store in the dictionary
    kernelDictionary[n] = impulseResponse[:trim]
    # Normalize
    kernelDictionary[n] /= np.sqrt( np.sum( kernelDictionary[n]**2 ) )
  return kernelDictionary