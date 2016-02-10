"""
tonality.py - Matrix representations of musical scores, corpara, and their tonality

Example: J. S. Bach's "Well Tempered Clavier" Books 1 and 2

2015, Michael A. Casey, Dartmouth College, Bregman Media Labs

License: 
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
http://creativecommons.org/licenses/by-nc/4.0/
"""

import pylab as P
import numpy as np
import glob, sys, pdb
import bregman

try:
    import music21 as m21
except:
    print "Warning: music21 not installed, only loading .ascii files supported"

pc_labels = np.tile(['C','C#','D','Eb','E','F','F#','G','G#','A','Bb','B'],13)

def _report(reportStr):
    print reportStr
    sys.stdout.flush()

def extract_audio_chroma(flist, nSecs = 10, nSamps = 6):
    """
    Given a list of WAV files, extract chromagram features and stack
    Options:
        nSecs - how many seconds each sample lasts
        nSamps - how many samples to take from each file        
    """
    F = []
    for fname in flist:
        x,sr,fmt = bregman.sound.wavread(fname)
        if len(x.shape) > 1: # Check if stereo
            x = x.mean(1) # Convert stereo to MONO
        for _ in range(nSamps):
            start = np.random.randint(x.shape[0] - sr*nSecs) # 10s segments    
            y = x[start:start+nSecs*sr]
            chrom = bregman.features.Chromagram(y, nfft=8192, wfft=8192, nhop=sr/20)
            X = chrom.X.T.reshape(nSecs,-1,12).mean(0).T # 1s averaging
            F.append(X)
    return np.hstack(F)

def load_corpus(corpus=None, idx=None, win_len=1, sample_len=0):
    """
    Load items from a corpus, use given idx slice argument to select subsets
    Inputs:
         corpus - list of symbolic music files (xml, mid, krn, etc...) 
            idx - slice argument giving range of works [None] (all)
        win_len - num tactus beats to integrate [1] (no integration)
     sample_len - number of sampled windows per work [0] (all)
    """
    if corpus is None or (type(corpus) is str and corpus is not ""):
        corpus_path=glob.glob(m21.__path__[0]+'/corpus/bach/bwv8[4-9][0-9]')
        corpus = []
        for w in corpus_path:
            for v in sorted(glob.glob(w+'/*')):
                corpus.append(v)
    corpus.sort()
    print idx
    _report("slicing work list...")
    idx = slice(0,len(corpus)) if idx is None else idx    
    _report("parsing corpus...")
    corpusList = [m21.converter.parse(w) for w in corpus[idx]]
    _report("extracting notes...")
    corpusNotes = [_extract_notes_positions_and_durations(w) for w in corpusList]
    _report("converting notes to matrices...")
    corpusMtx = [_sample_mtx(_convert_notes_to_matrix(n), win_len, sample_len) for n in corpusNotes]
    _report("done.")
    return corpusMtx

def _sample_mtx(M, win_len, sample_len):
    if win_len>1:
        M = win_mtx(M, win_len)
    if sample_len>0:
        M = M[:,np.random.permutation(M.shape[1])[:sample_len]]
    return M

def _extract_notes_positions_and_durations(work):
    """
    Return note positions and durations
    """
    notes = np.array([(nn.midi,n.offset,n.quarterLength) for n in work.flat.notes for nn in n.pitches])
    notes = notes[np.where(notes[:,2])]
    return notes

#edit to include manual length and smallest duration
# start_t is start time in quarter notes
# duration is duration in quarter notes
def _convert_notes_to_matrix(notes, start_t=0, duration=128): # start_t and duration offset in quarters
    """
    Given a list of (midi,quarterLength) tuples, collate all notes per tactus tick (smallest duration) and
    make piano-roll matrix
    """
    smallest_dur = _calc_smallest_dur(notes) #manually calculate if none given
    start_times = np.array(notes)[:,1] # 
    time_idx = (start_times >= start_t) & (start_times < start_t + duration)
    notes = np.array(notes).copy()[time_idx]
    t0 = notes[0,1]
    N = notes[-1,1] - t0
    d = notes[-1,2]
    Nc = (N+d) / smallest_dur
    mtx = np.zeros((128,Nc))
    for n in notes:
        mtx[n[0],(n[1]-t0)/smallest_dur:(n[1]-t0+n[2])/smallest_dur]=1
    return mtx

#calculate smallest interval
def _calc_smallest_dur(notes):
    tick = np.array(notes)[:,2].min()
    return tick

def load_wtc(idx=None, win_len=1, sample_len=0):
    """
    Load scores in matrix form in the entire WTC dataset.
    Inputs:
            idx - slice argument giving range of works [None] (all)
        win_len - num tactus beats to integrate [1] (no integration)
     sample_len - number of sampled windows per work [0] (all)
    """
    flist = sorted(glob.glob('*.ascii'))
    if idx is not None:
        if not np.iterable(idx):
            idx = [idx]
    else:
        idx = range(len(flist))
    flist = np.array(flist)[idx]
    if win_len>0:
        A = [win_mtx(np.loadtxt(fname, dtype='i4'),win_len) for fname in flist]
    else:
        A = [np.loadtxt(fname, dtype='i4').mean(1) for fname in flist]
    if win_len>0 and sample_len>0:
        AA = [a[:,np.random.permutation(a.shape[1])[:sample_len]] for a in A]
    else:
        AA = A
    return AA

def win_mtx(a, win_len=2):
    """
    Options:
        win_len  - window length [2]
    """
    # perform simple integration
    N = np.ceil(a.shape[1]/float(win_len))
    aa = []
    for k in np.arange(N-1):
        aa.append(a[:,k*win_len:(k+1)*win_len].mean(1)) 
    return np.vstack(aa).T

def fold_mtx(a):
    """
    Fold piano-roll matrix into single octave beginning with 'C'.
    """
    return a[:120,:].reshape(-1,12,a.shape[1]).mean(0)

def dissimilarity_mtx(A):
    """
    Given a piano-roll indicator matrix, construct self-dissimilarity matrix
    """
    D = bregman.distance.euc_normed(A.T,A.T)
    return D

def center_mtx(D):
    """
    Given a dissimilarity or dissonance matrix, center the matrix by subtracting the mean of 
    the rows and columns. For a dissimilarity matrix this operation yields the "scatter matrix".
    """
    H = np.eye(D.shape[0]) - 1.0/D.shape[0]
    B = np.dot(np.dot(H,-0.5*D),H)
    return B

def dissonance_mtx(A):
    """
    Given a piano-roll indicator matrix, construct pair-wise dissonance matrix    
    """
    n = A.shape[1]
    D = np.zeros((n,n))
    for i,a in enumerate(A.T[:-1]):
        for j,b in enumerate(A.T[i+1:]):
            D[i,j] = dissonance_fun(np.expand_dims(a+b,1))

def dissonance_fun(A):
    """
    Given a piano-roll indicator matrix representation of a musical work (128 pitches x beats),
    return the dissonance as a function of beats.
    Input:
        A  - 128 x beats indicator matrix of MIDI pitch number

    """
    freq_rats = np.arange(1,11) # Harmonic series ratios
    amps = np.exp(-.5 * freq_rats) # Partial amplitudes
    F0 = 8.1757989156 # base frequency for MIDI (note 0)
    diss = [] # List for dissonance values
    thresh = 1e-3
    for beat in A.T:
        idx = np.where(beat>thresh)[0]
        if len(idx):
            freqs, mags = [], [] # lists for frequencies, mags
            for i in idx:
                freqs.extend(F0*2**(i/12.0)*freq_rats)
                mags.extend(amps)
            freqs = np.array(freqs)
            mags = np.array(mags)
            sortIdx = freqs.argsort()
            d = _dissonance_fun(freqs[sortIdx],mags[sortIdx])
            diss.extend([d])
        else:
            diss.extend([-1]) # Null value
    return np.array(diss)


def _dissonance_fun(freqs, amps=None, params=None):
    """
    ::
        Compute dissonance between partials with center frequencies in freqs,
        and amplitudes in amps. Based on William Sethares after Plomp and Levelt:

        default params = (-3.51, -5.75, 0.0207, 19.96, 5, -5, 0.24)
        default amps, use 1 as amplitude for all partials.
    """
    if params is None: params = (-3.51, -5.75, 0.0207, 19.96, 5, -5, 0.24)
    b1, b2, s1, s2, c1, c2, Dstar  = params
    if amps is None: amps = [1]*len(freqs)
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

def plot_mtx(mtx=None, title=None, newfig=False, cbar=True, **kwargs):
    """
    ::

        static method for plotting a matrix as a time-frequency distribution (audio features)
    """
    if mtx is None or type(mtx) != np.ndarray:
        raise ValueError('First argument, mtx, must be a array')
    if newfig: P.figure()
    dbscale = kwargs.pop('dbscale', False) 
    bels = kwargs.pop('bels',False)
    norm = kwargs.pop('norm',False)
    normalize = kwargs.pop('normalize',False)
    origin=kwargs.pop('origin','lower')
    aspect=kwargs.pop('aspect','auto')
    interpolation=kwargs.pop('interpolation','nearest')
    cmap=kwargs.pop('cmap',P.cm.gray_r)
    clip=-100.
    X = scale_mtx(mtx, normalize=normalize, dbscale=dbscale, norm=norm, bels=bels)
    i_min, i_max = np.where(X.mean(1))[0][[0,-1]]
    X = X[i_min:i_max+1].copy()
    if dbscale or bels:
        if bels: clip/=10.
        P.imshow(P.clip(X,clip,0),origin=origin, aspect=aspect, interpolation=interpolation, cmap=cmap, **kwargs)
    else:
        P.imshow(X,origin=origin, aspect=aspect, interpolation=interpolation, cmap=cmap, **kwargs)
    if title:
        P.title(title,fontsize=16)
    if cbar:
        P.colorbar()
    P.yticks(np.arange(0,i_max+1-i_min,3),pc_labels[i_min:i_max+1:3],fontsize=14)
    P.xlabel('Tactus', fontsize=14)
    P.ylabel('MIDI Pitch', fontsize=14)
    P.grid()

def scale_mtx(M, normalize=False, dbscale=False, norm=False, bels=False):
    """
    ::

        Perform mutually-orthogonal scaling operations, otherwise return identity:
          normalize [False]
          dbscale  [False]
          norm      [False]        
    """
    if not (normalize or dbscale or norm or bels):
        return M
    else:
        X = M.copy() # don't alter the original
        if norm:
            nz_idx = (X*X).sum(1) > 0
            X[nz_idx] = (X[nz_idx].T / np.sqrt((X[nz_idx]*X[nz_idx]).sum(1))).T
        if normalize:
            X=X-np.min(X)
            X=X/np.max(X)
        if dbscale or bels:
            X = P.log10(P.clip(X,0.0001,X.max()))
            if dbscale:                
                X = 20*X
    return X

def hist_mtx(mtx, tstr=''):
    """
    Given a piano-roll matrix, 128 MIDI piches x beats, plot the pitch class histogram
    """
    i_min, i_max = np.where(mtx.mean(1))[0][[0,-1]]
    P.figure(figsize=(14.5,8))    
    P.stem(np.arange(i_max+1-i_min),mtx[i_min:i_max+1,:].sum(1))
    ttl = 'Note Frequency'
    if tstr: ttl+=': '+tstr
    P.title(ttl,fontsize=16)
    t=P.xticks(np.arange(0,i_max+1-i_min,3),pc_labels[i_min:i_max+1:3],fontsize=14)
    P.xlabel('Pitch Class', fontsize=14)
    P.ylabel('Frequency', fontsize=14)
    ax = P.axis()
    P.axis(xmin=-0.5)
    P.grid()

if __name__ == "__main__":
    P.interactive(True)
    a = np.loadtxt('01.ascii')
    P.figure()
    # Plot piano roll: MIDI pitch by beats
    P.subplot(211)
    plot_mtx(a, cmap=P.cm.gray_r, cbar=False)
    P.axis('tight')
    P.title('WTC 1 "Prelude in C": Piano Roll')

    # Plot dissonance by (integrated) beats
    P.subplot(212)
    win_len=8 # Number of beats to integrate, non-overlapping
    a = win_mtx(a, win_len)
    d = dissonance_fun(a)
    P.plot(np.arange(len(d))*win_len, d,'r',linewidth=1)
    P.axis('tight')
    P.title('Dissonance (win_len=%d)'%win_len, fontsize=16)

