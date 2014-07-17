# AudioCollection - manage audio collections and databases
# Bregman - python toolkit for music information retrieval

__version__ = '1.0'
__author__ = 'Michael A. Casey'
__copyright__ = "Copyright (C) 2010  Michael Casey, Dartmouth College, All Rights Reserved"
__license__ = "gpl 2.0 or higher"
__email__ = 'mcasey@dartmouth.edu'


import os
import os.path
import time
import glob
import tempfile
import shutil
import subprocess
import hashlib
import random
import error
import features

try: # OSX / Linux
    BREGMAN_ROOT = os.environ['HOME']
except: # Windows
    BREGMAN_ROOT = os.environ['HOMEPATH']

DATA_ROOT = BREGMAN_ROOT + os.sep + "exp"

class AudioCollection:
    """
    ::

        A class for extracting, persisting, and searching audio features in a collection

        Initialization:
          AudioCollection("/path/to/collection")

        Instantiate a new collection at the given path. This directory is where features and
        audioDB databases will be stored. The collection path does not have to be the same as the
        audio files path for inserted audio. Audio can come from any location, but features are consolodated into
        the AudioCollection path.
    """

    collection_stem = "collection_"

    def __init__(self, path=None, root_path=DATA_ROOT):
        self.root_path = root_path 
        self.collection_path = path
        self.adb_path = None
        self.adb = None
        self.rTupleList = None
        self.feature_params = features.Features.default_feature_params()
        self.audio_collection = set()
        self.cache_temporary_files = False
        self.uid="%016X"%long(random.random()*100000000000000000L) # unique instance ID
        self.old_uid=self.uid
        self.adb_data_size=256
        self.adb_ntracks=20000

    def insert_audio_files(self, audio_list):
        """
        ::

            Maintain a set of audio files as a collection. Each item is unique in the set. Collisions are ignored.
        """
        for item in audio_list: self.audio_collection.add(item)
        
    def _gen_adb_hash(self):
        """
        ::

            Generate a hash key based on self.feature_params to make an audioDB instance unique to feature set.
        """
        m = hashlib.md5()
        k = self.feature_params.keys()
        k.sort() # ensure vals ordered by key lexicographical order
        vals = [self.feature_params.get(i) for i in k]
        m.update(vals.__repr__())
        return m.hexdigest()

    def _insert_features_into_audioDB(self):
        """
        ::

            Persist features, powers, and audio-file keys in an audioDB instance.
            Inserts self.rTupleList into audioDB instance associated with current feature set.
            If features already exist, warn once, but continue.
            Given an adb-instance path, exectute the audioDB command to:
                Make lists of features, powers, and database keys
                Batch insert features, powers, linked to databse keys
                Returns full path to audioDB instance if OK or None of not OK
        """
        self._new_adb() # test to see if we require a new audiodb instance for features
        pth, sep, nm = self.adb_path.rpartition(os.sep)
        # List names for audioDB insertion
        fListName = pth + os.sep + self.uid + "_fList.txt"
        pListName = pth + os.sep + self.uid + "_pList.txt"
        kListName = pth + os.sep + self.uid + "_kList.txt"
        # unpack the names of files we want to insert
        fList, pList, kList = zip(*self.rTupleList)
        # write fList, pList, kList to text files
        self._write_list_to_file(fList, fListName)
        self._write_list_to_file(pList, pListName)
        self._write_list_to_file(kList, kListName)
        # do BATCHINSERT
        command = ["audioDB", "--BATCHINSERT", "-d", self.adb_path, "-F", fListName, "-W", pListName, "-K", kListName]
        self._do_subprocess(command)
        return 1

    def _write_list_to_file(self,lst, pth):
        """
        ::

            Utility routine to write a list of strings to a text file
        """
        try:
            f = open(pth,"w")
        except:
            print "Error opening: ", pth
            raise error.BregmanError()
        for s in lst: f.write(s+"\n") 
        f.close()

    def extract_features(self, key=None, keyrepl=None, extract_only=False, wave_suffix=".wav"):
        """
        ::

            Extract features over the collection
            Pre-requisites:
             self.audio_collection - set of audio files to extract
             self.feature_params - features to extract

            Arguments:
             key - optional string to append on filename stem as database key
             keyrepl - if key is specified then keyrepl is a pattern to replace with key
             extract_only - set to True to skip audioDB insertion
             wave_suffix - fileName extension for key replacement

            Returns rTupleList of features,powers,keys or None if fail.
        """
        aList, fList, pList, kList = self._get_extract_lists(key, keyrepl, wave_suffix)
        self._fftextract_list(zip(aList,fList,pList,kList))
        self.rTupleList = zip(fList,pList,kList) # what will be inserted into audioDB
        if not extract_only:
            self._insert_features_into_audioDB() # possibly generate new audiodb instance
        self.audio_collection.clear() # clear the audio_collection queue
        return self.rTupleList

    def _get_extract_lists(self, key=None, keyrepl=None, wave_suffix=".wav"):
        """
        ::

            Map from self.audio_collection to aList, fList, pList, kList
        """
        # The audio queue
        aList = list(self.audio_collection)
        aList.sort()
        # The keys that will identify managed items
        if key == None:
            kList = aList # use the audio file names as keys
        else:
            # replace keyrepl with key as database key
            if not keyrepl: 
                print "key requires keyrepl for filename substitution"
                raise error.BregmanError()
            kList = [a.replace(keyrepl,key) for a in aList]
        feature_suffix= self._get_feature_suffix()
        power_suffix=self.feature_params['power_ext']
        fList = [k.replace(wave_suffix, feature_suffix) for k in kList]
        pList = [k.replace(wave_suffix, power_suffix) for k in kList]
        return (aList, fList, pList, kList)

    def _get_feature_suffix(self):
        """
        ::
        
            Return a standardized feature suffix for extracted features
        """
        return "." + self.feature_params['feature'] + "%02d"%self.feature_params['ncoef']

    def _fftextract_list(self, extract_list):
        command=[]
        feature_keys = {'stft':'-f', 'cqft':'-q', 'mfcc':'-m', 'chroma':'-c', 'power':'-P', 'hram':'-H'}
        feat = feature_keys[self.feature_params['feature']]
        ncoef = "%d"%self.feature_params['ncoef']
        nfft = "%d"%self.feature_params['nfft']
        wfft = "%d"%self.feature_params['wfft']
        nhop = "%d"%self.feature_params['nhop']
        logl = "%d"%self.feature_params['log10']
        mag = "%d"%self.feature_params['magnitude']
        lo = "%f"%self.feature_params['lo']
        hi = "%f"%self.feature_params['hi']
#        lcoef = "%d"%self.feature_params['lcoef'] # not used yet        
        for a,f,p,k in extract_list:
            if not len(glob.glob(f)):
                command=["fftExtract", "-p", "bregman.wis", 
                         "-n", nfft, "-w", wfft, "-h", nhop, feat, ncoef, 
                         "-l", lo, "-i", hi, "-g" , logl, "-a", mag, a, f]
                ret = self._do_subprocess(command)
                if ret:
                    print "Error extacting features: ", command
                    return None
            else:
                print "Warning: feature file already exists", f
            if not len(glob.glob(p)):
                command=["fftExtract", "-p", "bregman.wis", 
                         "-n", nfft, "-w", wfft, "-h", nhop, 
                         "-P", "-l", lo, "-i", hi, a, p]
                ret = self._do_subprocess(command)
                if ret:
                    print "Error extacting powers: ", command
                    return None
            else:
                print "Warning: power file already exists", p

    def _remove_temporary_files(self, key=""):
        """
        ::

            Remove cached feature and power files based on current feature_params settings.        
        """        
        fList = glob.glob(self.collection_path+os.sep + "*" + key + "." 
                             + self.feature_params['feature']+"%02d"%self.feature_params['ncoef'])
        for f in fList: os.remove(f)
        pList = glob.glob(self.collection_path + os.sep + "*" + key + self.feature_params["power_ext"] )
        for p in pList: os.remove(p)

    def initialize(self):
        """
        ::

            Make a new collection path with an empty audioDB instance.
            Each instance is unique to a set of feature_params.
            Return False if an equivalent instance already exists.
            Return True if new instance was created.
        """
        if not self._gen_collection_path(self.collection_stem): 
            return 0
        print "Made new directory: ", self.collection_path
        # self._new_adb() # This is now done on feature_insert
        return 1

    def _gen_adb_path(self):
        """
        ::

            Name a new adb instance
        """
        if not self.collection_path:
            print "Error: self.collection_path not set"
            raise error.BregmanError()
        adb_path = self.collection_path + os.sep + self.collection_stem + self._gen_adb_hash() +".adb"
        return adb_path
        
    def _gen_collection_path(self, name_prefix):
        """
        ::

            Make a new unique directory in the self.root_path directory
        """
        self.collection_path = tempfile.mkdtemp(prefix=name_prefix,dir=self.root_path)
        if not self.collection_path:
            print "Error making new directory in location: ", self.root_path
            return 0
        shutil.copymode(self.root_path, self.collection_path) # set permissions
        return 1

    def _new_adb(self):
        """
        ::

            Make a new audioDB instance in the adb_path location
            Make database L2norm and Power compliant
        """
        self.adb_path = self._gen_adb_path()
        if self.adb_path == None:
            print "self.adb_path must have a string value"
            raise error.BregmanError()
        else:
            f = None
            try:
                f = open(self.adb_path,"rb")
            except:
                print "Making new audioDB database: ", self.adb_path
            finally:
                if f:
                    f.close()
                    print "AudioDB database already exists: ", self.adb_path
                    return 0

        # create a NEW audiodb database instance
        command = ["audioDB", "--NEW", "-d", self.adb_path, "--datasize", "%d"%self.adb_data_size, "--ntracks", "%d"%self.adb_ntracks]
        self._do_subprocess(command)

        # make L2NORM compliant
        command = ["audioDB", "--L2NORM", "-d", self.adb_path]
        self._do_subprocess(command)

        # make POWER compliant
        command = ["audioDB", "--POWER", "-d", self.adb_path]
        self._do_subprocess(command)
        return 1

    def _do_subprocess(self,command):
        """
        ::

            Call an external (shell) command, inform about any errors
        """
        res = subprocess.call(command)
        if res:
            print "Error in ", command
            raise error.BregmanError()
        return res

    def load(self, path=None):
        """
        ::

            Load stored data for this collection
        """
        pass

    def save(self):
        """
        ::

            Save data for this collection
        """
        pass

    def toc(self, collection_path=None):
        """
        ::

            List contents of this collection, or collection at collection_path.
        """
        if collection_path == None:
            collection_path = self.collection_path
        dlist=glob.glob(collection_path + os.sep + "*.data")
        toclist = []
        for d in dlist:
            self.load(d)
            toclist.append(self.feature_params)        
        return zip(dlist,toclist)

    @classmethod
    def lc(cls, expand=False, limit=None):
        """
        ::

            Alias for ls_collections()
        """
        return cls.ls_collections(expand, limit)

    @classmethod
    def ls_collections(cls, expand=False, limit=None):
        """
        ::

            For the given class, return a list of instances
            If expand is set to True, pint each collection's TOC
        """
        dlist, tm = cls._get_collection_list_by_time()
        dlist = zip(dlist[:limit], tm[:limit])
        if expand:
            R = cls()
            k = R.toc(dlist[0][0])
            for d,t in dlist: 
                print d, t
                print k[0][1].keys()
                for i, v in enumerate(R.toc(d)):
                    print "[%d]"%i, v[1].values()
                print ""
        return dlist

    @classmethod
    def _get_collection_list_by_time(cls):
        dlist = glob.glob(DATA_ROOT + os.sep + cls.collection_stem + "*")
        # sort into descending order of time
        tm = {}
        for d in dlist: tm[ d ] = os.path.getmtime( d )
        dlist.sort( lambda x,y: cmp( tm[x], tm[y] ) )
        dlist.reverse()
        tm = [time.ctime(tm[d]) for d in dlist]
        return (dlist, tm)

