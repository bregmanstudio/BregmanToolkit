# Bregman - python toolkit for music information retrieval

__version__ = '1.0'
__author__ = 'Michael A. Casey'
__copyright__ = "Copyright (C) 2010  Michael Casey, Dartmouth College, All Rights Reserved"
__license__ = "gpl 2.0 or higher"
__email__ = 'mcasey@dartmouth.edu'

# Exception Handling class
class BregmanError(Exception):
    def __init__(self, msg):
        print "Bregman error:", msg
