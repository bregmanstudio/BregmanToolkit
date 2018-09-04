# tuning.py - computation of tuning systems in Python
# Author: Michael A. Casey
# Copyright (C) January 2011, Dartmouth College, All Rights Reserved

# Bregman - python toolkit for music information retrieval

__version__ = '1.0'
__author__ = 'Michael A. Casey'
__copyright__ = "Copyright (C) 2010  Michael Casey, Dartmouth College, All Rights Reserved"
__license__ = "GPL Version 2.0 or Higher"
__email__ = 'mcasey@dartmouth.edu'


from fractions import Fraction
from .psychoacoustics import dissonance_fun
from .testsignal import default_signal_params, harmonics
import pylab
import pdb

class TuningSystem():
    """
    ::

        A class for generating tuning systems, scales, and intervals.

        Usage examples:
          T = TuningSystem()
          print T.EQ # equal-temperament 12-tet frequency ratios
          print T.PY # Pythagorean frequency ratios
          print T.JI # just intonation frequency ratios
          print T.BP # equal temperament Bohlen-Pierce 13-step tritave ratios
          eq = T.to_scale_freqs(T.JI, f0=220) # realize the scale frequencies starting at 220Hz
          x = T.to_scale_sound(T.EQ, f0=220, num_harmonics=6) # realize scale as sound
          x = T.to_scale_intervals(T.PY, f0=220, interval=7, num_harmonics=6) # realize scale as intervals with p5
          sound(x*0.05, 44100)

        The methods equal_temperament(), Pythagorean(), and just_intonation(), can generate the scales
        with different parameterizatoins, such as the order (number of tuning steps) to use.
    """
    def __init__(self):
        pitch_class_sharp = ['U', 'm2', 'M2', 'm3', 'M3', 'P4', 'TT', 'P5', 'm6', 'M6', 'm7', 'M7', 'P8']
        pitch_class_flat =  ['M7', 'm7', 'M6', 'm6', 'P5', 'TT', 'P4', 'M3', 'm3', 'P2', 'm2', 'U']
        pitch_class_flat.reverse()
        self.pitch_class = pitch_class_sharp
        self.pitch_ratios_sharp = None
        self.pitch_ratios_flat = None
        self.reference_frequency = 440.
        self.EQ = self.equal_temperament()
        self.PY = self.Pythagorean()
        self.JI = self.just_intonation()
        self.BP = self.Bohlen_Pierce()

    def equal_temperament(self, frame_interval=2, num_steps=12):
        """
        ::

            Defines 12-tet pitch classes in Equal Temperament as irrational numbers.
        """

        self.pitch_ratios_eq = self.generalized_equal_temperament(frame_interval, num_steps)
        return self.pitch_ratios_eq

    def Bohlen_Pierce(self, frame_interval=3, num_steps=13):
        """
        ::

            Defines 13-tet pitch classes in Bohlen-Pierce tritave as irrational numbers.
        """
        self.pitch_ratios_bp = self.generalized_equal_temperament(frame_interval, num_steps)
        return self.pitch_ratios_bp

    def generalized_equal_temperament(self, frame_interval=2, num_steps=12):
        """
        ::

            Defines 12-tet pitch classes in Equal Temperament as irrational numbers.
        """
        f = self.reference_frequency
        self.pitch_ratios_sharp = [Fraction(k,num_steps) for k in range(num_steps+1)]
        self.pitch_ratios_flat = [Fraction(-num_steps+k,num_steps) for k in range(num_steps)]
        self.pitch_ratios_get = [frame_interval**k for k in self.pitch_ratios_sharp]
        return self.pitch_ratios_get
       
    def Pythagorean(self, N=12):
        """
        ::

            Generate 12 steps of Pythagorean tuning as rational numbers
            from N cycles in P5 and P4 directions.
        """
        ratios = []
        self.pitch_ratios_sharp = [0]*N
        self.pitch_ratios_flat = [0]*N
        for k in range(N):
            ratios.append(Fraction(3,2)**k)
            while ratios[k] > 2:
                ratios[k] /= 2
        for r in range(N): self.pitch_ratios_sharp[(r*7)%12] = ratios[r%12]

        ratios = []
        for k in range(1,N+1):
            ratios.append(Fraction(2,3)**k)
            while ratios[k-1] < 1:
                ratios[k-1] *= 2
        for r in range(N): self.pitch_ratios_flat[((r+1)*5)%N]=ratios[r]
        self.pitch_ratios_pythagorean = self.sort_ratios(self.pitch_ratios_flat + self.pitch_ratios_sharp)
        return self.pitch_ratios_pythagorean
    
    def just_intonation(self, N=25):
        """
        ::

            Generate 12 steps of Just Intonation tuning from N harmonics
        """
        ratios = []
        self.pitch_ratios_sharp = [0]*N
        self.pitch_ratios_flat = [0]*N
        for k in range(N):
            ratios.append(Fraction(k+1,1))
            while ratios[k] > 2:
                ratios[k] /= 2
        self.pitch_ratios_sharp = ratios
        self.pitch_ratios_just = self.sort_ratios(self.pitch_ratios_sharp)
        return self.pitch_ratios_just

    def sort_ratios(self, ratios):
        """
        ::

            Sort and select tuning system ratios in ascending order by
            proximity to equal tempered pitch classes.
        """
        ratios.append(Fraction(2,1))
        res = list(set(ratios)) # eliminate duplicates
        res.sort() # in-place sort
        idx = [pylab.argmin( abs( pylab.array(res) - j) ) for j in self.pitch_ratios_eq]
        res = [res[i] for i in idx]
        return res

    def to_scale_freqs(self, tuning, f0=440.):
        """
        ::

            Convert tuning system list to a scale frequencies at f0 [default=440.Hz]
        """
        rt = [float(r) * f0 for r in tuning]
        return rt

    def to_dissonance(self, tuning, f0=440., num_harmonics=6):
        """
        ::

            Convert scale to dissonance values for num_harmonics harmonics.
            Assume an exponentially decaying harmonic series.
            Returns dissonance scores for each interval in given tuning system.
        """
        harm = pylab.arange(num_harmonics)+1
        h0 = [f0 * k for k in harm]
        a = [pylab.exp(-0.5 * k) for k in harm]
        diss = [dissonance_fun(h0 + [p*k for k in harm]) for p in self.to_scale_freqs(tuning, f0)]
        return diss

    def to_scale_sound(self, tuning, f0=440., num_harmonics=6, dur=0.5, sr = 44100.):
        """
        ::

            Realize given tuning system as a scale, starting at frequency
            f0=440., using num_harmonics=6 complex tones with duration dur=0.5 seconds, and
            default sample rate sr=44100Hz.
        """
        p = default_signal_params()
        p['num_harmonics'] = num_harmonics
        p['sr'] = sr
        p['num_points']=int(round(dur*sr))
        p.pop('f0')
        x = []
        num_samples = p['num_points']
        ramp = pylab.r_[pylab.array(pylab.linspace(0,1,100)), pylab.linspace(1,0,num_samples-100)]
        for rat in tuning:
            x = pylab.r_[x, ramp*harmonics(f0=f0*rat,**p)]
        return x

    def to_scale_intervals(self, tuning, interval=0, f0=440., num_harmonics=6, dur=0.5, sr = 44100.):
        """
        ::

            Realize given tuning system as a series of intervals, with
            scale degree interval=0 (tonic), starting at frequency
            f0=440., using num_harmonics=6 complex tones with duration
            dur=0.5 seconds, and default sample rate sr=44100Hz.
        """
        dummy = [tuning[interval]]*len(tuning)
        x = self.to_scale_sound(dummy, f0=f0, num_harmonics=num_harmonics, dur=dur, sr=sr)
        y = self.to_scale_sound(tuning, f0=f0, num_harmonics=num_harmonics, dur=dur, sr=sr)
        return (x + y)/2
        
