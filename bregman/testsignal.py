#testsignal.py - generate some multi-dimensional test signals
#
#Author: Michael A. Casey
#Copyright (C) 2010 Bregman Music and Audio Research Studio
#Dartmouth College, All Rights Reserved
#
# A collection of test signal generators
#

# Bregman - python toolkit for music information retrieval

__version__ = '1.0'
__author__ = 'Michael A. Casey'
__copyright__ = "Copyright (C) 2010  Michael Casey, Dartmouth College, All Rights Reserved"
__license__ = "gpl 2.0 or higher"
__email__ = 'mcasey@dartmouth.edu'

import pylab
import scipy.signal

TWO_PI = 2.0 * pylab.pi

# Exception Handling class
class TestSignalError(Exception):
    """
    Test signal exception class.
    """
    def __init__(self, msg):
        print "TestSignal: " + msg

# Return parameter dict used by all of the test signal generators
def default_signal_params():
    """    
    ::
    
      Return a new parameter dict consisting of:
        'f0':441.0,
        'num_points':44100,
        'num_harmonics':1,
        'phase_offset':0.0,
        'sr':44100.0
    """
    p = {
        'f0':441.0,
        'num_points':44100,
        'num_harmonics':1,
        'phase_offset':0.0,
        'sr':44100.0
        }
    return p

def _check_signal_params(params):
    """
    ::
      
      check signal params for missing keys and insert default values
    """
    p = default_signal_params()
    params = p if params == {} else params.copy()
    params['f0'] = float(params.get('f0', p['f0']))
    params['sr'] = float(params.get('sr', p['sr']))
    params['phase_offset'] = float(params.get('phase_offset', p['phase_offset']))
    params['num_harmonics'] = int(params.get('num_harmonics', p['num_harmonics']))
    params['num_points'] = int(params.get('num_points', p['num_points']))
    return params

# A single sinusoid
def sinusoid(**params):
    """
    ::

        Generate a sinusoidal audio signal from signal_params: see default_signal_params() 
          **params - signal_params dict, see default_signal_params()

    """
    params = _check_signal_params(params)
    t = pylab.arange(params['num_points'])
    x = pylab.sin( TWO_PI*params['f0']/params['sr'] * t + params['phase_offset'])
    return x

# Harmonic sinusoids
def harmonics(afun=lambda x: pylab.exp(-0.5*x), pfun=lambda x: pylab.rand()*2*pylab.pi,**params):
    """
    ::

        Generate a harmonic series using a harmonic weighting function
         afun   - lambda function of one parameter (harmonic index) returning a weight
         pfun   - lambda function of one parameter (harmonic index) returning radian phase offset
         **params - signal_params dict, see default_signal_params()
    """
    params = _check_signal_params(params)
    f0 = params['f0']
    x = pylab.zeros(params['num_points'])
    for i in pylab.arange(1, params['num_harmonics']+1):    
        params['f0']=i*f0
        params['phase_offset'] = pfun(i)
        x +=  afun(i) * sinusoid(**params)
    x = balance_signal(x,'maxabs')
    return x

# Shepard tones from harmonic sinusoids
def shepard(num_octaves=7, center_freq=440, band_width=150,**params):
    """
    ::

        Generate shepard tones
             num_octaves - number of sinusoidal octave bands to generate [7]
             center_freq - where the peak of the spectrum will be [440]
             band_width - how wide a spectral band to use for shepard tones [150]
             **params - signal_params dict, see default_signal_params()
    """
    params = _check_signal_params(params)
    f0 = params['f0']
    x = pylab.zeros(params['num_points'])
    shepard_weight = gauss_pdf(20000, center_freq, band_width)
    for i in pylab.arange(num_octaves):
        a = shepard_weight[int(round(f0*2**i))]
        params['f0']=f0*2**i
        x += a * harmonics(**params)
    x = balance_signal(x,'maxabs')
    return x

# 1d Gaussian kernel
def gauss_pdf(n,mu=0.0,sigma=1.0):
    """
    ::

        Generate a gaussian kernel
         n - number of points to generate
         mu - mean
         sigma - standard deviation
    """
    var = sigma**2
    return 1.0 / pylab.sqrt(2 * pylab.pi * var) * pylab.exp( -(pylab.r_[0:n] - mu )**2 / ( 2.0 * var ) )

# Chromatic sequence of shepard tones
def devils_staircase(num_octaves=7, num_steps=12, step_size=1, hop=4096, 
                     overlap=True, center_freq=440, band_width=150,**params):
    """
    ::

        Generate an auditory illusion of an infinitely ascending/descending sequence of shepard tones
            num_octaves - number of sinusoidal octave bands to generate [7]
            num_steps - how many steps to take in the staircase
            step_size - semitone change per step, can be fractional [1.]
            hop - how many points to generate per step [12]
            overlap - whether the end-points should be cross-faded for overlap-add
            center_freq - where the peak of the spectrum will be [440]
            band_width - how wide a spectral band to use for shepard tones [150]
            **params - signal_params dict, see default_signal_params()

    """
    params = _check_signal_params(params)
    sr = params['sr']
    f0 = params['f0']
    norm_freq = 2*pylab.pi/sr
    wlen = min([hop/2, 2048])
    x = pylab.zeros(num_steps*hop+wlen)
    h = scipy.signal.hanning(wlen*2)
    # overlap add    
    params['num_points']=hop+wlen
    phase_offset=0
    for i in pylab.arange(num_steps):
        freq = f0*2**(((i*step_size)%12)/12.0)
        params['f0'] = freq
        s = shepard(num_octaves=num_octaves, center_freq=center_freq, band_width=band_width, **params)
        s[0:wlen] = s[0:wlen] * h[0:wlen]
        s[hop:hop+wlen] = s[hop:hop+wlen] * h[wlen:wlen*2]
        x[i*hop:(i+1)*hop+wlen] = x[i*hop:(i+1)*hop+wlen] + s
        phase_offset = phase_offset + hop*freq*norm_freq
    if not overlap:
        x = pylab.resize(x, num_steps*hop)
    x = balance_signal(x,'maxabs')
    return x

# Overlap-add two signals
def overlap_add(x, y, wlen):
    """
    ::

        Overlap-add two sequences x and y by wlen samples
    """
    z = pylab.zeros(x.size + y.size - wlen)
    z[0:x.size] = x;
    z[x.size-wlen:x.size+y.size-wlen]+=y
    return z

# Parameter dict for noise test signals
def default_noise_params():
    """
    ::

        Returns a new parameter dict for noise generators consisting of:
             'noise_dB':24.0  - relative amplitude of noise to harmonic signal content
             'num_harmonics':1 - how many harmonics (bands) to generate
             'num_points':44100, - how many points to generate
             'cf':441.0 - center frequency in Hertz
             'bw':50.0 - bandwidth in Hertz
             'sr':44100.0 - sample rate in Hertz
    """
    p = {'noise_dB':24.0,
         'num_harmonics':1,
         'num_points':44100,
         'cf':441.0,
         'bw':50.0,
         'sr':44100.0
         }
    return p

def _check_noise_params(params):
    """
    ::
    
      test for dict keys and insert default values for missing keys
    """
    p = default_noise_params()
    params = p if params is None else params.copy()
    params['noise_dB'] = float(params.get('noise_dB',p['noise_dB']))
    params['cf'] = float(params.get('cf',p['cf']))
    params['bw'] = float(params.get('bw',p['bw']))
    params['sr'] = float(params.get('sr',p['sr']))
    params['num_points'] = int(params.get('num_points',p['num_points']))
    params['num_harmonics'] = int(params.get('num_harmonics',p['num_harmonics']))
    return params

# Combine harmonic sinusoids and noise signals
def noise(noise_fun=pylab.rand,**params):
    """
    ::

        Generate noise according to params dict
            params - parameter dict containing sr, and num_harmonics elements [None=default_noise_params()]
            noise_fun - the noise generating function [pylab.rand]
    """
    params = _check_noise_params(params)
    noise_dB = params['noise_dB']
    num_harmonics = params['num_harmonics']
    num_points = params['num_points']
    cf = params['cf']
    bw = params['bw']
    sr = params['sr']
    g = 10**(noise_dB/20.0) * noise_fun(num_points)
    [b,a] = scipy.signal.filter_design.butter(4, bw*2*pylab.pi/sr, btype='low', analog=0, output='ba')
    g = scipy.signal.lfilter(b, a, g)
    # Phase modulation with *filtered* noise (side-band modulation should be narrow-band at bw)
    x = pylab.sin( (2.0*pylab.pi*cf / sr) * pylab.arange(num_points) + g)
    return x

def modulate(sig, env, nsamps):
    """
    ::

        Signal modulation by an envelope
        sig - the full-rate signal
        env - the reduced-rate envelope
        nsamps - audio samples per envelope frame
    """
    if( sig.size != len(env)*nsamps ):
        print "Source signal size must equal len(env) * nsamps"
        return False
    y = pylab.zeros(sig.size)
    start = 0
    for a in env:
        end = start + nsamps
        y[start:end] = a * sig[start:end]
        start = end
    return y

def default_rhythm_params():
    """
    ::

        Return signal_params and pattern_params dicts, and a patterns tuple for 
        a default rhythm signal such that:
                'sr' : 48000,        # sample rate
                'bw' : [80., 2500., 1000.], # band-widths
                'cf' : [110., 5000., 16000.], # center-frequencies
                'dur': [0.5, 0.5, 0.5] # relative duration of timbre
                'amp': [1.0, 1.0, 1.0] # relative amplitudes of timbres
                'normalize' : 'maxabs' # balance timbre channels 'rms', 'max', 'maxabs', 'norm', 'none'
        Example:
         signal_params, rhythm_params, patterns = default_rhythm_params()
         sig = rhythm(signal_params, rhythm_params, patterns)
    """
    sp = {
        'sr' : 48000,
        'tc' : 2.0,
        'cf' : [110., 5000., 16000.],
        'bw' : [80., 2500., 1000.],
        'dur' : [1.0, 0.5, 0.25],
        'amp': [1.0, 1.0, 1.0],
        'normalize' : 'maxabs'
        }
    rp = {
        'tempo' : 120.,
        'subdiv' : 16
        }
    pats = (0b1010001010100000, 0b0000100101001001, 0b1010101010101010)
    return (sp, rp, pats)

def _check_rhythm_params(signal_params, rhythm_params, patterns):
    s,r,p = default_rhythm_params()
    for k in s.keys(): # check for missing keys
        signal_params[k] = signal_params.get(k, s[k])
    for k in r.keys(): # check for missing keys
        rhythm_params[k] = rhythm_params.get(k, r[k])    
    num_timbres = len(signal_params['cf'])
    if not ( num_timbres == len(signal_params['bw']) == len(signal_params['dur']) == len(patterns) ):
        return 0
    if not num_timbres: 
        raise TestSignalError("rhythm: signal_params lists and pattern n-tuple lengths don't match")
    return num_timbres

def balance_signal(sig, balance_type="maxabs"):
    """
    ::
    
        Perform signal balancing using:
          rms - root mean square
          max - maximum value
          maxabs - maximum absolute value
          norm - Euclidean norm
          none - do nothing [default]
        
        Returns:
          sig - balanced (normalized) signal
    """
    balance_types = ['rms', 'max', 'maxabs', 'norm', 'none']
    if balance_type==balance_types[0]:
        return sig / pylab.rms_flat(sig)
    if balance_type==balance_types[1]:
        return sig / sig.max()    
    if balance_type==balance_types[2]:
        return sig / abs(sig).max()    
    if balance_type==balance_types[3]:
        return sig / pylab.norm_flat(sig)
    if balance_type==balance_types[4]:
        return sig    
    raise TestSignalError("signal balancing type not supported: %s"%balance_type)

def rhythm(signal_params=None, rhythm_params=None, patterns=None):
    """
    ::

        Generate a multi-timbral rhythm sequence using noise-band timbres 
        with center-frequency, bandwidth, and decay time controls

        Timbre signal synthesis parameters are specified in 
        the signal_params dict:
            ['cf'] - list of center-frequencies for each timbre
            ['bw'] - list of band-widths for each timbre
            ['dur'] - list of timbre durations relative to a quarter note
            ['amp'] - list of timbre relative amplitudes [default 1.0]
            ['sr'] - sample rate of generated audio
            ['tc'] - constant of decay envelope relative to subdivisions:
             The following expression yields a time-constant for decay to -60dB 
             in a given number of beats at the given tempo:
               t = beats * tempo / 60.
               e^( -tc * t ) = 10^( -60dB / 20 )
               tc = -log( 0.001 ) / t           

        The rhythm sequence is generated with musical parameters specified in
        the rhythm_params dict: 
            ['tempo']  - how fast
            ['subdiv'] - how many pulses to divide a 4/4 bar into

        Rhythm sequences are specified in the patterns tuple (p1,p2,...,pn)
           patterns - n-tuple of integers with subdiv-bits onset patterns, 
            one integer element for each timbre

           Parameter constraints:
             Fail if not:
               len(bw) == len(cf) == len(dur) == len(patterns)
    """
    # Short names
    p = default_rhythm_params()
    signal_params = p[0] if signal_params is None else signal_params
    rhythm_params = p[1] if rhythm_params is None else rhythm_params
    patterns = p[2] if patterns is None else patterns
    num_timbres = _check_rhythm_params(signal_params, rhythm_params, patterns)

    # convenience variables
    sp = signal_params
    rp = rhythm_params

    # Duration parameters
    qtr_dur = 60.0 / rp['tempo'] * sp['sr'] # duration of 1/4 note
    eth_dur = 60.0 / (2.0 * rp['tempo']) * sp['sr'] # duration of 1/8 note
    sxt_dur = 60.0 / (4.0 * rp['tempo']) * sp['sr'] # duration of 1/16 note
    meter = 4.0
    bar_dur = meter * qtr_dur # duration of 1 bar

    # Audio signal wavetables from parameters
    ns_sig=[]
    ns_env=[]
    for cf, bw, dur, amp in zip(sp['cf'], sp['bw'], sp['dur'], sp['amp']):
        ns_par = default_noise_params()
        ns_par['sr'] = sp['sr']
        ns_par['cf'] = cf
        ns_par['bw'] = bw
        ns_par['num_points'] = 2 * bar_dur 
        ns_sig.append(amp * noise(**ns_par))
        ns_env.append( pow( 10, -sp['tc'] * pylab.r_[ 0 : 2 * bar_dur ] / (qtr_dur * dur) ) )

    # Music wavetable sequencer
    snd = [[] for _ in range(num_timbres)] 
    snd_ptr = [qtr_dur  for _ in range(num_timbres)]
    num_beats = rp['subdiv']
    test_bit = 1 << ( num_beats - 1 )
    dt = 16.0 / num_beats
    for beat in range(num_beats):
        for p, pat in enumerate(patterns):
            if (pat & (test_bit >> beat) ): snd_ptr[p] = 0

        for t in range(num_timbres):
            idx = pylab.array(pylab.r_[snd_ptr[t]:snd_ptr[t]+sxt_dur*dt], dtype='int')
            snd[t].append( ns_sig[t][idx] * ns_env[t][idx] )
            snd_ptr[t] += sxt_dur * dt

    all_sig = pylab.concatenate( snd[0] )
    for t in pylab.arange(1, num_timbres):
        sig = pylab.concatenate( snd[t] )
        all_sig += sig
    return balance_signal(all_sig, sp['normalize'])


