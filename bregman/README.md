BregmanToolkit
==============
<pre>
Audio and Music Analysis and Synthesis in Python

Author: Michael A. Casey
Copyright (C) 2010-2011 Trustees of Dartmouth College, All Rights Reserved
License: GPL 2.0 or later

Quick Start Guide
ipython # start ipython environment in a shall

from bregman.suite import *

p = default_feature_params()
audio_file = os.path.join(audio_dir,"gmin.wav")
F = Features(audio_file, p)
imagesc(F.X,dbscale=True)
title('Default constant-Q spectrogram')

F.inverse(F.X, pvoc=True) # invert features to audio
play(balance_signal(F.x_hat),F.sample_rate)

p['feature']='stft'
p['nfft']=1024
p['wfft']=512
p['nhop']=256
F = Features(audio_file, p)
imagesc(F.X,dbscale=True)
title('Wide-band spectrogram')

F.inverse(F.X) # invert features to audio
play(balance_signal(F.x_hat),F.sample_rate)

# Feature built-in tutorial examples
tuts = get_tutorials()
execfile(tuts[0])

# Separation (using PLCA) built-in tutorial examples
execfile(tuts[1])

</pre>
