import os,sys,glob
import pylab
import bregman
import os.path

from .features import *
from .segment import *
from .audiodb import *
from .testsignal import *
from .psychoacoustics import *
from .tuning import *
from .sound import *
from .plca import *
from .distance import *
from .classifier import *
from .error import *
from .beat import *

from . import features
from . import segment
from . import testsignal
from . import psychoacoustics
from . import tuning
from . import sound
from . import plca
from . import distance
from . import classifier
from . import error
from . import beat

sep = os.path.sep
bregman_data_root = os.path.split(bregman.__file__)[0]
examples_dir = os.path.join(bregman_data_root,"examples"+sep)
audio_dir = os.path.join(bregman_data_root,"audio"+sep)
sys.path.append(examples_dir)

def get_tutorials():
    """
    print and return a list of tutorials in the bregman/examples folder
    """
    tlist = glob.glob(os.path.join(examples_dir,"*.py"))
    tlist.sort()
    return tlist



