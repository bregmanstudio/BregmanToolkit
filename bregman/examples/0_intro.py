# Getting started with the bregman toolkit
from bregman.suite import *

print "BREGMAN>We're going to use our built-in audio examples in audio_dir"
print "audio_file = os.path.join(audio_dir,'gmin.wav')"
audio_file = os.path.join(audio_dir,"gmin.wav")

print "\nBREGMAN>Perform a short-time Fourier transform, specify windowing parameters"
print "linspec = LinearFrequencySpectrum(audio_file, nfft=1024, wfft=512, nhop=256)"
print "linspec.feature_plot(dbscale=True)"
print "title('Wide-band Linear Spectrum')"
# Short-time Fourier transform
linspec = LinearFrequencySpectrum(audio_file, nfft=1024, wfft=512, nhop=256)
linspec.feature_plot(dbscale=True)
title('Wide-band Linear Spectrum')

print "\nBREGMAN>Next, play the audio_file using the built-in play() command"
print "x,sr,fmt = wavread(audio_file) # load the audio file"
print "play(balance_signal(x),sr) # play it"
print "[Press any key to continue...]"
waitforbuttonpress(60)
x,sr,fmt = wavread(audio_file) # load the audio file
play(balance_signal(x),sr) # play it


print "\nBREGMAN>Now we'll invert the short-time Fourier transform using the feature inverse() method"
print "x_hat = linspec.inverse(usewin=0) # invert features to audio (use original phases, no windowing)"
print "play(balance_signal(x_hat))"
print "[Press any key to continue...]"
waitforbuttonpress(60)
x_hat = linspec.inverse(usewin=0) # invert features to audio (use original phases, no windowing)
play(balance_signal(x_hat))

print "\nBREGMAN>Next, extract a Log-frequency spectrum, specifying windowing parameters"
print "logspec = LogFrequencySpectrum(audio_file, nhop=2205) # extract log spectrum"
print "logspec.feature_plot(dbscale=True) # plot features on dB scale"
print "title('Narrow-band Log Spectrum')"
print "[Press any key to continue...]"
waitforbuttonpress(60)

logspec = LogFrequencySpectrum(audio_file, nhop=2205) # extract log spectrum
logspec.feature_plot(dbscale=True) # plot features on dB scale
title('Narrow-band Log Spectrum')

print "[Press any key to continue...]"
waitforbuttonpress(60)

print "\nBREGMAN>Now we'll invert the log spectrum using the feature inverse() method"
print "x_hat = logspec.inverse(pvoc=True) # invert phaseless features to audio"
print "play(balance_signal(x_hat),sr) # play inverted features"
print "[Press any key to continue...]"
waitforbuttonpress(60)
x_hat = logspec.inverse(pvoc=True) # invert phaseless features to audio
play(balance_signal(x_hat),sr) # play inverted features

print "\nBREGMAN>List the (default) parameters that control feature extraction."
print "Use any parameter as a keyword argument to a feature extractor."
print "Or use a parameter dict {'key1':value1, ...} with the base Feature class"
print "[Press any key to continue...]"
waitforbuttonpress(60)
p = Features.default_params() # inspect default parameters
for parameter in p: print parameter+': ', p[parameter] # show feature extraction parameter dict

print "\nBREGMAN>Next, use built-in help to view the pydoc help for the features module"
print "[Press any key to continue...]"
waitforbuttonpress(60)
help(features) # see help on the features module 

print "\nBREGMAN>That's the intro. Try execfile on the remaining tutorials:"
tuts = get_tutorials()
for tutorial in tuts: print tutorial


