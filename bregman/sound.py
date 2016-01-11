#sound.py - audio file I/O and play functionality
# Bregman - python toolkit for music information retrieval

__version__ = '1.0'
__author__ = 'Michael A. Casey'
__copyright__ = "Copyright (C) 2010  Michael Casey, Dartmouth College, All Rights Reserved"
__license__ = "GPL Version 2.0 or Higher"
__email__ = 'mcasey@dartmouth.edu'

import os
import numpy
import subprocess
import error
import pdb

# audio file handling
try:
    import scikits.audiolab
    HAVE_AUDIOLAB=True
    from scikits.audiolab import *
except ImportError:
    HAVE_AUDIOLAB=False
import wave 

# WaveOpen class
class WavOpen:
    """
    ::

        WavOpen: sound-file handling class
        wav = WavOpen(filename,n)
        filename - sound file to open
        n - chunk size per read [0=all frames]
    """        
    def __init__(self,arg,n=None, verbosity=1):
        self.sound=None
        self.open(arg, n, verbosity)

    def __iter__(self):
        return self

    def open(self,filename, n, verbosity=0):
        if self.sound is None:
            self.filename = filename
            self.sound = wave.open(self.filename, "r")
            self.index = 0
            self.sample_rate = self.sound.getframerate()
            if verbosity:
                print 'sample_rate=%i' %self.sample_rate
            self.num_channels = self.sound.getnchannels()
            if verbosity:
                print 'num_channels=%i' %self.num_channels
            self.sample_width = self.sound.getsampwidth()
            if verbosity:
                print 'sample_width=%i' %self.sample_width
            self.num_frames = self.sound.getnframes()
            if verbosity:
                print 'num_frames=%i, num_secs=%f' %(self.num_frames, self.num_frames/self.sample_rate)
            self.n = self.num_frames if n is None else n
            self.buffer_size = min(self.n, 16384)
            self.bytes_per_frame = self.sample_width * self.num_channels
            if verbosity:
                print 'bytes_per_frame=%i' %self.bytes_per_frame
            self.bytes_per_second = self.sample_rate * self.bytes_per_frame
            if verbosity:
                print 'bytes_per_second=%i' %self.bytes_per_second
            self.bytes_per_buffer = self.buffer_size * self.bytes_per_frame
            if verbosity:
                print 'bytes_per_buffer=%i' %self.bytes_per_buffer
            self.sig = numpy.zeros((self.n,), dtype='float32')
        else:
            self.rewind()

    def close(self):
        if self.sound is not None:
            self.sound.close()
            self.sound = None
        else:
            raise Exception("Sound already closed")

    def rewind(self):
        if self.sound is not None:
            self.close()
            self.sound=None
            self.open(self.filename,self.n)

    def get_all_frames(self):
        rawdata = self.sound.readframes(self.num_frames)
        if rawdata:
            signal = wave.struct.unpack('%ih' %self.num_frames*self.num_channels, rawdata) # transform to signal
            del rawdata
            self.sig = numpy.zeros((self.num_frames,), dtype='float32')
            for index in range(self.num_frames):
                self.sig[index] = signal[index*self.num_channels] / 32768.

    def __len__(self):        
        return self.num_frames / self.n

    def __getitem__(self,k):
        while k < 0:
            k += self.num_frames/self.n
        if not k < self.num_frames/self.n:
            raise IndexError
        self.sound.setpos(k*self.n)
        self.index = k*self.n
        return self.next()

    def next(self):
        num_to_read = self.n if self.index < (self.num_frames - self.n + 1) else (self.num_frames - self.index)
        if num_to_read > 0:
            rawdata = self.sound.readframes(num_to_read)
            self.sig[:] = 0
            if rawdata:
                signal = wave.struct.unpack('%ih' %(num_to_read*self.num_channels), rawdata) # transform to signal
                for index in range(num_to_read):
                    self.sig[index] = signal[index*self.num_channels] / 32768.
                self.index += num_to_read
                return self.sig[::]
            else:
                raise Exception("Could not read sufficient sound data at index %d"%self.index)            
        else:
            raise StopIteration

        
def wav_write(signal, wav_name, sample_rate):
    """
    ::

        Utility routine for writing wav files, use scikits.audiolab if available
        otherwise uses wave module
    """
    return _wav_write(signal, wav_name, sample_rate)

def _wav_write(signal, wav_name, sample_rate):
    """
    ::

        Utility routine for writing wav files, use scikits.audiolab if available
    """
    if HAVE_AUDIOLAB:
        scikits.audiolab.wavwrite(signal, wav_name, sample_rate)
    else:
        signal = numpy.atleast_2d(signal)
        w = wave.Wave_write(wav_name)
        if not w:
            print "Error opening file named: ", wav_name
            raise error.BregmanError()
        w.setparams((signal.shape[0],2,sample_rate,signal.shape[1],'NONE','NONE'))
        b_signal = '' # C-style binary string
        for i in range(signal.shape[1]):
            b_signal += wave.struct.pack('h',int(32767*signal[0,i])) # transform to C-style binary string
            if signal.shape[0]>1:
                b_signal += wave.struct.pack('h',int(32767*signal[1,i])) # transform to C-style binary string            
        w.writeframes(b_signal)
        w.close()
    return True

def wav_read(wav_name):
    """
    ::

        Utility routine for reading wav files, use scikits.audiolab if available
        otherwise uses wave module.
    """
    return _wav_read(wav_name)

def _wav_read(wav_name):
    """
    ::

        Utility routine for reading wav files, use scikits.audiolab if available
        otherwise uses wave module.
    """
    if HAVE_AUDIOLAB:
        signal, sample_rate, pcm = scikits.audiolab.wavread(wav_name)
        return (signal, sample_rate)
    else:
        wav=WavOpen(wav_name)
        wav.get_all_frames()
        return (wav.sig, wav.sample_rate)

# Define sound player helper functions
AUDIO_TMP_FILE = ".tmp.wav"
sound_options = {"soundplayer": "open"}

# Bregman's own play_snd(...) function
if os.name=='posix':
    def play_snd(data, sample_rate=44100):
        """
        ::

            Bregman Linux/OSX/Windows sound player function.
            data - a numpy array
            sample_rate - default 44100
        """
        m = abs(data).max() + 0.001
        if  m > 1.0: data /= m
        _wav_write(data, AUDIO_TMP_FILE, sample_rate)
        command = [sound_options['soundplayer'], AUDIO_TMP_FILE]
        res = subprocess.call(command)
        if res:
            raise error.BregmanError("Error in "+command[0])
        return res            
else:  
    import winsound
    def play_snd(data, sample_rate=44100):            
        """
        ::

            Bregman Linux/OSX/Windows sound player function.
            data - a numpy array
            sample_rate - default 44100
        """
        m = abs(data).max() + 0.001
        if  m > 1.0: data /= m
        _wav_write(data, AUDIO_TMP_FILE, sample_rate)
        winsound.PlaySound(AUDIO_TMP_FILE, winsound.SND_FILENAME|winsound.SND_ASYNC)

# Emulate the play(), wavread, and wavwrite functions from audiolab
if not HAVE_AUDIOLAB:
    def play(data, fs=44100):
        """
        Wrapper function for Bregman's play_snd(data, sample_rate) using system audioplayer.
        Install scikits.audiolab for native Python sound playback
        """
        play_snd(data, fs)

    def wavread(filename):
        """
        Wrapper function for Bregman's wav_read(filename) using python built-ins.
        Install scikits.audiolab for more efficient sound I/O

        Returns:
          data [ndarray]
          sample_rate [int]
          fmt ['pcm16']
        """
        x,sr = wav_read(filename)
        return x,sr,'pcm16'
        
    def wavwrite(signal, wav_name, sample_rate):
        """
        Wrapper function for Bregman's wav_write(data, filename, sample_rate) using python built-ins.
        Install scikits.audiolab for more efficient sound I/O

        Returns:
           True - if successful
        """
        return wav_write(signal, wav_name, sample_rate)

