
from bregman.suite import *
from scipy.signal import lfilter
from pylab import zeros, where, diff, sqrt


def beat_track(x, feature=LogFrequencySpectrum, **kwargs):
    """
    Scheirer beat tracker. Use output of comb filter bank on filterbank
                           sub-bands to estimate tempo, and comb filter state
                           to track beats.
    inputs:
       x        - the audio signal or filename to analyze
       feature  - the feature class to use [LogFrequencySpectrum]
       **kwargs - parameters to the feature extractor [nbpo=1, nhop=441]
    outputs:
       z      - per-tempo comb filter summed outputs
       tempos - the range of tempos in z
       D      - the differentiated half-wave rectified octave-band filterbank
                outputs
    """
    kwargs.setdefault('nhop', 441)
    kwargs.setdefault('nbpo', 1)
    F = feature(x, **kwargs)
    frame_rate = F.sample_rate / float(F.nhop)
    D = diff(F.X, axis=1)
    D[where(D < 0)] = 0
    tempos = list(range(40, 200, 4))
    z = zeros((len(tempos), D.shape[0]))
    for i, bpm in enumerate(tempos):  # loop over tempos to test
        t = int(round(frame_rate * 60. / bpm))  # num frames per beat
        alpha = 0.5**(2.0/t)
        b = [1 - alpha]
        a = zeros(t)
        a[0] = 1.0
        a[-1] = alpha
        z[i, :] = lfilter(b, a, D).sum(1)  # filter and sum sub-band onsets
    return z, tempos, D
