# examples_soundspotter.py
#
# An example application using the bregman toolkit
#
# Match segments from a target audio file to segments from a source database
# using overlapping sequences audio features (shingles)
#
# Based on the following publications:
#
# Casey, M. "Soundspotter", http://sourceforge.net/projects/mp7c/ (2003-)
#
# Casey, M. and Slaney, M. "The Importance of Sequences for Music Similarity", 
# Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 
# Toulouse, France, May 2006
#
# Casey, M., "Soundspotting: A New Kind of Process?" 
# in R. Dean (Ed.), The Oxford Handbook of Computer Music and Digital Sound Culture, 
# Oxford University Press, 2009.
#
# Author: Michael A. Casey
# Bregman Music and Auditory Research Studio, Dartmouth College

from bregman.suite import *
from bisect import bisect_right
from numpy import concatenate, hstack, cumsum, array, zeros, exp, hamming, isnan
from matplotlib.mlab import rms_flat

default_params = default_feature_params()
default_params['feature']='mfcc'
default_params['ncoef']=20
default_params['nhop']=2205

def extract_target_feature_sequences(target_file, p=default_params, win=20, hop=10):
    """ 
    inputs:
     target_file - file name of target media (the one to reconstruct)
     p - feature_params (see default_params)
     win - stack features into length win sequences
     hop - skip hop vectors per query
    returns:
     features X - feature matrix for target
    """
    F = Features(target_file,p)
    X = adb.stack_vectors(F.X.T, win, hop).T
    return X

def _feature_list_to_bounds(y_list):
    # convert list of features to bounds array for mapping global match locations 
    # to per-media locators
    bounds = [0]
    for y in y_list:
        bounds.append(y.shape[1])
    return cumsum(array(bounds))

def extract_source_feature_sequences(source_files, p=default_params, win=20, hop=10):
    """
    inputs:
     source_files - list of file names of source media (the database)
     p - feature_params (see default_params)
     win - stack features into length win sequences
     hop - skip hop vectors per query
    returns:
     features Y - concatenated source feature matrices
         bounds - list of cumulative lengths of matrices
     """
    features = list()
    for s in source_files:
        F = Features(s,p)
        features.append(adb.stack_vectors(F.X.T, win, hop).T)
    return features

def match_sequences(X, Y, num_hits=1):
    """
    make a list of matches per target vector
     inputs:
      X - the target features (column-major)
      Y - the list of source features (column-major)      
      num_hits - how many matches to retrieve per query
     returns:
      matches - array of match positions, num_hits per input vector
      distances - array of distances, num_hits per input vector
    """
    D = euc_normed(X.T,hstack(Y).T)
    DD = D.argsort(axis=1)[:,:num_hits]
    D = array([D[k,DD[k,:]] for k in range(D.shape[0])])
    return DD, D

def _bounds_to_locator(m, bounds):
    #return index of media corresponding to global match position in bounds
    return bisect_right(bounds,m)-1

def _bounds_to_index(m, bounds):
    # return media locator corresponding to global match position in bounds
    return m - bounds[_bounds_to_locator(m,bounds)]

def _fetch_audio(f, i, p, w, h):
    # retrieve audio segment from file
    try:
        x,sr,fmt = wavread(f, first=i*h*p['nhop'], last=w*p['nhop'])
        x = x.sum(1) if len(x.shape)>1 else x
    except RuntimeError: # not enough samples at end of file
        x,sr,fmt = wavread(f, first=i*h*p['nhop'], last=None)
        x = x.sum(1) if len(x.shape)>1 else x
        x = concatenate([x,zeros(w*p['nhop']-len(x))]) # zero pad incomplete frame
    return x

def _sequence_overlap_add(y_list, p, win, hop):
    # make new signal by overlapping and adding list of signals
    y = zeros((len(y_list)*hop+win-1)*p['nhop'])
    for i,k in enumerate(range(0, len(y_list)*hop*p['nhop'], hop*p['nhop'])):
        end_k = min(len(y), k + win*p['nhop'])
        y[k:end_k] += y_list[i][:end_k-k]
    return y

def reconstruct_audio(matches, distances, bounds, target_file, source_files, p, win, hop, beta=2.0):
    """
    make a new audio signal based on matches and source media 
     inputs:
      matches - list of matches from match_sequences
      distances - list of distances from match_sequences
      target_file - file name of target media (the one to reconstruct)      
      source_files - list of file names of source media (the database)
      p - feature parameters
      win - sequence length
      hop - sequence hop
      beta - stiffness coefficient for mixing based on distances [2.0]
     returns:
      y - the reconstructued audio signal
    """
    y_list = list()
    hamm = hamming(p['nhop']*2)[:p['nhop']]
    for i in range(len(matches)):
        x = _fetch_audio(target_file, i, p, win, hop)
        y = zeros((win*p['nhop']))
        for j, m in enumerate(matches[i,:]):
            yy = _fetch_audio(source_files[_bounds_to_locator(m,bounds)], _bounds_to_index(m,bounds), p, win, hop)
            y +=  yy * exp(-beta * distances[i,j]) # weight match contribution by distance prior
        y *= rms_flat(x) / rms_flat(y) # energy balance output rms using input rms
        if win>1 and hop<win:
            y[:p['nhop']]*=hamm
            y[:-p['nhop']-1:-1]*=hamm
        y_list.append(y) 
    return _sequence_overlap_add(y_list, p, win, hop)

def soundspotter(target_file, source_files, p=default_params, win=20, hop=10, num_hits=1, beta=2.0, X=None, Y=None):
    """
    A soundspotter
     returns new audio based on target_file using source_files
     inputs:
       target - name of target media
       sources - list of file names of source media
       p - feature parameter dict [Features.defalt_params()]
       win - sequence of vectors size for temporal context [20]
       hop - number of vectors to advance per query [10]
       num_hits - number of matches to find per query vector [1]
       beta - stiffness coefficient for mixing based on distances [2.0]
          X - pre-computed features for target
          Y - list of pre-computed features for sources
     outputs:
       y - new audio signal
    """
    X = X if X is not None else extract_target_feature_sequences(target_file, p, win, hop)
    Y = Y if Y is not None else extract_source_feature_sequences(source_files, p, win, hop)
    bounds = _feature_list_to_bounds(Y)
    matches, distances = match_sequences(X,Y,num_hits)
    y = reconstruct_audio(matches,distances,bounds,target_file,source_files,p,win,hop,beta)
    y[isnan(y)]=0
    return y    


if __name__ == "__main__":
    print "Running soundspotter on audio directory..."
    default_params['nhop']=882
    default_params['lcoef']=3
    sources = glob.glob(os.path.join(audio_dir,"*.wav"))
    sources.sort()
    y = soundspotter(os.path.join(audio_dir,"amen.wav"), sources[1:],
                     p=default_params, win=5, hop=4, num_hits=5, beta=5.0)
    play(balance_signal(y,'maxabs'))

