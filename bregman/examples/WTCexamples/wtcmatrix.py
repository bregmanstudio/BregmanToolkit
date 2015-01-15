"""
 wtcmatrix - convert list of scores into matrix form
 
 Requires:
     Music21 version 1.4.0+      - web.mit.edu/music21/
     BregmanToolkit              - https://github.com/bregmanstudio/BregmanToolkit

2015, Michael A. Casey, Dartmouth College, Bregman Media Labs

License: 
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
http://creativecommons.org/licenses/by-nc/4.0/
"""

from pylab import array, roll, ones, zeros, dot, std, mean, sort, sqrt, where, imshow, figure, cm, arange, xticks, yticks, grid,mod, kron, r_, c_, plot
import music21 as m21
import glob, pdb, sys, os
import bregman
from scipy.signal import resample

def report(msg):
    print(msg)
    sys.stdout.flush()

def load_wtc(idx=None, corpus=m21.__path__[0]+'/corpus/bach/bwv8[4-9][0-9]'):
    """
    Load items from a corpus, use given idx slice argument to select subsets
    """
    wtc = glob.glob(corpus)
    wtc.sort()
    idx = slice(0,len(wtc)) if idx is None else idx
    WTC = []
    for w in wtc[idx]:
        for v in sort(glob.glob(w+'/*')):
            WTC.append(m21.converter.parse(v))
    return WTC

def get_notes_and_durations_by_measure(work_list):
    """
    Make a list of (midi, quarterLength) tuples per measure from list of works
    """
    notes = [[[(nn.midi,n.quarterLength) for n in w.measure(k).flat.notes for nn in n.pitches] 
          for k in xrange(1,len(w.measureOffsetMap())+1)] for w in work_list]
    return notes

def get_notes_and_durations_by_quarters(work_list, num_qtrs=2):
    """
    Make a list of (midi, quarterLength) tuples per quaterLength from list of works
    """
    notes = [[[(nn.midi,n.quarterLength) for n in w.flat.notes.getElementsByOffset(i,i+num_qtrs,includeEndBoundary=False) for nn in n.pitches] 
         for i in xrange(0,int(max([o['offset'] for o in w.flat.notes.offsetMap]))+num_qtrs,num_qtrs)] for w in work_list]
    return notes

def extract_notes_positions_and_durations(work_list):
    """
    Return note positions and durations
    """
    notes = [[(nn.midi,n.offset,n.quarterLength) for n in w.flat.notes for nn in n.pitches] for w in work_list]
    na = array(notes[0])
    notes = na[where(na[:,2])]
    #pdb.set_trace()
    return [notes]

#edit to include manual length and smallest duration
# start_t is start time in quarter notes
# duration is duration in quarter notes
def convert_notes_to_matrix(notes_list, start_t=0, duration=128): # start_t and duration offset in quarters
    """
    Given a list of (midi,quarterLength) tuples, collate all notes per tactus tick (smallest duration) and
    make piano-roll matrix
    """
    mtx_list = []
    for nl in notes_list: #where does nl come from?
        smallest_dur = _calc_smallest_dur(nl) #manually calculate if none given
        start_times = array(nl)[:,1] # 
        time_idx = (start_times >= start_t) & (start_times < start_t + duration)
        nl = array(nl).copy()[time_idx]
        t0 = nl[0,1]
        N = nl[-1,1] - t0
        d = nl[-1,2]
        Nc = (N+d) / smallest_dur
        mtx = zeros((128,Nc))
        for n in nl:
            mtx[n[0],(n[1]-t0)/smallest_dur:(n[1]-t0+n[2])/smallest_dur]=1
        mtx_list.append(mtx)
    return mtx_list

#calculate smallest interval
def _calc_smallest_dur(nl):
    tick = array(nl)[:,2].min()
    return tick

def plot_mtx(m, beats=4,tick=0.25, **kwargs):
    """
    Plot piano-roll matrix
    """
    figure()
    kwargs.setdefault('cmap',cm.ocean_r)
    imshow(m,aspect='auto',origin='bottom',**kwargs)
    nr,nc = m.shape
    xt = arange(0,nc,beats/tick)
    xticks(xt,arange(1,len(xt)+1))
    grid(axis='x',linestyle='--')    
    pc=['C','C#','D','Eb','E','F','F#','G','G#','A','Bb','B']
    yt = arange(0,nr+1)
    yticks(yt,array(pc)[mod(yt,12)],fontsize=6)
    #grid(axis='y')
    staff_lines = array([4,7,11,14,17])
    staff_lines = array([staff_lines+12,staff_lines+36,staff_lines+60,staff_lines+84,staff_lines+108]).flatten()
    plot(c_[zeros(len(staff_lines)),nc*ones(len(staff_lines))].T,c_[staff_lines,staff_lines].T,'k')

def play_mtx(w, sr=32000, foffset=24, nhop=4410):
    """
    Invert a piano-roll matrix to audio 
    Return estimated signal
    """
    # inverse constant-Q transform
    F = bregman.features.LogFrequencySpectrum(bregman.testsignal.sinusoid(f0=441,num_points=44100),nbpo=12, nhop=4410)
    F.X = w[foffset:F.X.shape[0]+foffset,:]
    x_hat = F.inverse(pvoc=True)
    #bregman.sound.play(x_hat/x_hat.max(),sr)
    return x_hat

def convert_notes_to_signal(notes_list):
    """
    Generate an audible signal from a list of notes
    """
    sig = []
    tick=16
    for nn in notes_list:
        aa = array(nn)
        sig.append([])
        sig[-1] = zeros(aa[-1,1]*tick)
        sig[-1][array(aa[:,1]*16-1,'i4')]=2**(aa[:,0]/12.0)
        sig[-1] = sig[-1] - sig[-1][where(sig[-1])].mean()
        sig[-1] = resample(sig[-1], len(sig[-1])*8)
    return sig

