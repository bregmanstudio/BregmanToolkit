# examples_testsignal.py 
# Bregman audio testsignal synthesis examples
#
# Copyright (C) 2011 Mike Casey
# Dartmouth College, Hanover, NH
# All Rights Reserved
#

from bregman.suite import *
import os
import os.path
from pylab import *

def ex_1a():
    """
    sinusoid
    """
    print "Example 1: sinusoid"
    sys.stdout.flush()
    p = default_signal_params()
    x = sinusoid(p)
    play(x/x.max())

def ex_2a():
    """
    harmonics
    """    
    print "Example 2: harmonics"
    sys.stdout.flush()
    p = default_signal_params()
    p['num_harmonics'] = 7
    x = harmonics(p)
    play(x)

def ex_3a():
    """
    shepard tone
    """
    print "Example 3a: Shepard tones, Deutch's Tri-Tone Paradox"
    sys.stdout.flush()
    p = default_signal_params()
    p['num_harmonics']=7
    p['f0']=27.5
    x = shepard(p)
    p['f0']=27.5*2**(-0.5)
    y = shepard(p)
    play(r_[x,y])

def ex_3b():
    """
    gliding risset tones (devil's staircase)
    """
    print "Example 3b: Shepard tones as gliding Risset tones"
    sys.stdout.flush()
    p = default_signal_params()
    p['num_harmonics']=7
    p['f0']=27.5
    x = devils_staircase(p, num_octaves=5, num_steps=48, step_size=0.25, hop=4096)
    p['f0']=27.5*2**(-0.5)
    y = devils_staircase(p, num_octaves=5, num_steps=48, step_size=0.25, hop=4096)
    # play two gliding tones 1/2 octave apart
    play(0.5 * (x + y))

def ex_4a():
    """
    noise bands, bandwidth-expanided sinusoid
    """
    print "Example 4: noise band, bandwidth-expanded sinusoid"
    sys.stdout.flush()
    p = default_noise_params()
    x = noise(p)
    play(x)
    
def ex_5a():
    """
    rhythm from noise bands
    """
    print "Example 5: noise-band rhythm"
    sys.stdout.flush()
    s,r,p = default_rhythm_params()
    x = rhythm(s,r,p)
    play(x, s['sr'])

if __name__ == "__main__":

    ex_1a() # sinusoid

    ex_2a() # harmonics

    ex_3a() # shepard tone
    ex_3b() # gliding risset tones

    ex_4a() # noise band (bandwidth-expanded sinusoid)

    ex_5a() # noise-band percussive rhythm
    
