#!/usr/bin/env python
import glob
import os.path

from distutils.core import setup
from distutils.command.install import INSTALL_SCHEMES
for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib'] 

sep = os.path.sep

setup(name='Bregman',
      version='0.11-11.14',
      description='Bregman Music and Audio Python Toolkit',
      long_description="""This package provides tools for reading, analyzing, manipulating, storing, retrieving, viewing, and evaluating, information processing operations on audio files, including speech, music, and other mixed audio.""",

      author='Michael Casey',      
      author_email='mcasey [AT] dartmouth [DOT] edu',
      url='http://bregman.dartmouth.edu/',
      license='GPL v. 2.0 or higher',
      platforms=['OS X (any)', 'Linux (any)', 'Windows (any)'],
      packages=['bregman'],
      data_files=[('bregman'+sep+'audio'+sep, glob.glob('bregman'+sep+'audio'+sep+'*.wav')),
                  ('bregman'+sep+'examples'+sep, glob.glob('bregman'+sep+'examples'+sep+'*.py'))]
     )
