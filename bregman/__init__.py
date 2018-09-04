# Bregman - python toolkit for music information retrieval

__version__ = '1.0'
__author__ = 'Michael A. Casey'
__copyright__ = "Copyright (C) 2010  Michael Casey, Dartmouth College, All Rights Reserved"
__license__ = "GPL Version 2.0 or Higher"
__email__ = 'mcasey@dartmouth.edu'

__all__ = ["features", "segment", "audiodb", "testsignal", "psychoacoustics", "tuning", "sound", "plca", "distance", "classifier", "error", "beat", "suite"]

# import the bregman modules
from . import suite
from . import features
from . import segment
from . import audiodb
from . import testsignal
from . import psychoacoustics
from . import tuning
from . import sound
from . import plca
from . import distance
from . import classifier
from . import error
from . import beat


