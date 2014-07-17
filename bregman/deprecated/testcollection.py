# testcollection.py - automatic generation of music test sets, database instances, ground truth, and evaluators
#
#    Classes derived from the TestCollection base class do the following:
#     * Generate musical sequences via sequence parameters
#     * Automatically generate ground truth lists
#     * Generate audio test sets according to synthesis parameters
#     * Analyze audio collection via feature extractor parameters
#     * Manage audioDB database instances to store and search the audio features
#     * Evaluate multiple test sets using automatically generated ground truth
#     * Report on Evaluations in various formats: text output, file output, graphs
#
# Current test collection classes: 
#  RhythmTest - rhythmic sequences with envelope-modulated noise-band timbres
#
# Future test collection classes:
#  HarmonyTest - chord sequences with envelope-modulated harmonic timbres
#
# Copyright (C) 2010 Michael A. Casey
#  Bregman Music and Audio Research Studio, Dartmouth College, All Rights Reserved

# Bregman - python toolkit for music information retrieval

__version__ = '1.0'
__author__ = 'Michael A. Casey'
__copyright__ = "Copyright (C) 2010  Michael Casey, Dartmouth College, All Rights Reserved"
__license__ = "gpl 2.0 or higher"
__email__ = 'mcasey@dartmouth.edu'

import audiocollection
import sound
import audiodb
import testsignal

import pylab
import scipy.stats

import sys
import os
import glob
import pickle
import hashlib

# timbre-channel support
import plca
import classifier

# debugging support
import pdb

# Base keys for test data
BASE_KEY="%06d"%0  # 6 decimal digits
HASH_KEY="%032X"%0 # 32 hexadecmal digits
COUNT_KEY="%04X"%0 # 4 hexadecimal digits

# Exception Handling class
class TestCollectionError(Exception):
    def __init__(self):
        print "An error occured inside a function call"

class TestCollection(audiocollection.AudioCollection):
    """
    ::

        Base class for test collections, stub

        TestCollection objects encapsulate audio test sets, database instances, ground truth, and evaluators

        Classes derived from the TestCollection base class do the following:
         * Generate musical sequences via sequence parameters
         * Automatically generate ground truth lists
         * Generate audio test sets according to synthesis parameters
         * Analyze audio collection via feature extractor parameters
         * Manage audioDB database instances to store and search the audio features
         * Evaluate multiple test sets using automatically generated ground truth
         * Report on Evaluations in various formats: text output, file output, graphs
    """
    collection_stem = "testcollection_"
    def __init__(self, num_patterns=100, path=None, root_path=audiocollection.DATA_ROOT):
        audiocollection.AudioCollection.__init__(self, path, root_path)
        self.num_patterns=num_patterns
        self.patterns = None
        self.transformations = None
        self.seq_params = None
        self.syn_params = None
        self.syn_fun = None
        self.test_set_list = None
        self.ranks_list = None
        self.dists_list = None

    def initialize(self):
        """
        ::

            Generate a new TestCollection instance and ground-truth set
            Uses the current values in self.seq_params to make pattern set.
        """
        if self.collection_path and self.patterns:
            return True
        if not audiocollection.AudioCollection.initialize(self):
            return False
        self.gen_pattern_set(save_patterns=True)
        return True

    def gen_pattern_set(self, save_patterns=False):
        """
        ::

            Generate set patterns according to self.seq_params
            Write audio, extract features
            Persist in an audioDB (adb) database
        """
        # singleton: only one pattern set per TestCollection
        if not self._load_gt_patterns():
            print "Sequencing patterns..."
            self.patterns = self.sequence_patterns()
            if save_patterns:
                return self._save_patterns()
            else:
                return True
            
    def sequence_patterns(self):
        """
        ::

            Implement this method to generate ground-truth patterns from self.seq_params
        """
        pass

    def gen_transformation_set(self, save_transformations=False):
        """
        ::

            Generate transformations of self.patterns using seq_params
            Persist transformation set parameters via a pattern key
        """
        if self.collection_path == None:
            print "You must instantiate a ground-truth database first. Use get_pattern_set()"
            return
        if not self._transformation_set_exists(): 
            print "Transforming patterns..."
            self.transform_patterns()
            if save_transformations:
                self._save_transformations()
        else:
            self._load_transformations()

    def transform_patterns(self):
        """
        ::

            Implement this method to transform self.patterns to self.transformations via self.seq_params
        """
        pass

    def _transformation_set_exists(self):
        """
        ::

            Check to see if a transformation pattern set exists for this test collection instance.
            transformation patterns are unique to self.seq_params
            pattern keys are invertible via KeyMapper to recover parameters from database keys
            Return True if transformation pattern set exists.
            Return False if no transformation pattern set exists
        """
        key = self._gen_transformations_hash()
        if not key: return False
        f = None
        try:
            f = open(self.collection_path + os.sep + "transformations_" + key + ".patterns","r")
        except:
            return False
        finally:
            if f:
                f.close()
                return True

    def _save_patterns(self):
        """
        ::

            Persist the following data to this instance's collection_path directory:
               self.patterns - integer representation of each rhythm pattern
        """
        if self.collection_path==None:
            print "save_patterns requires self.collection_path"
            return False
        if self.patterns==None:
            print "save_patterns requires self.patterns"
            return False
        f = open(self.collection_path + os.sep + self.collection_stem + "groundtruth.patterns","w")
        pickle.dump(self.patterns, f)
        f.close()
        return True
    
    def _load_gt_patterns(self):
        """
        ::

            Retrieve the following data from this instance's collection_path directory:
               self.patterns - integer representation of each rhythm pattern
        """
        if self.collection_path==None:
            print "load_patterns requires self.collection_path"
            return
        f = None
        try:
            f = open(self.collection_path + os.sep + self.collection_stem + "groundtruth.patterns","r")
        except:
            return False
        self.patterns = pickle.load(f)
        f.close()
        return True

    def _save_transformations(self):
        """
        ::

            Persist the following data to this instance's collection_path directory:
               self.transformations - integer representation of each rhythm pattern
        """
        if self.collection_path==None:
            print "save_transformations requires self.collection_path"
            return False
        if self.transformations==None:
            print "save_transformations requires self.transformations"
            return False
        key = self._gen_transformations_hash()
        if not key:
            return False
        f = open(self.collection_path + os.sep + "transformations_" + key + ".patterns","w")
        pickle.dump(self.transformations, f)
        f.close()
        return True

    def _load_transformations(self, key=None):
        """
        ::

            Retrieve the following data to this instance's collection_path directory:
               self.transformations - integer representation of each rhythm pattern
        """
        if self.collection_path==None:
            print "load_transformations requires self.collection_path"
            return
        if not key:
            key = self._gen_transformations_hash()
        f = open(self.collection_path + os.sep + "transformations_" + key + ".patterns","r")
        self.transformations = pickle.load(f)
        f.close() 

    def _get_adb(self):
        if self.adb:
            return self.adb
        if self.adb_path:
            return audiodb.adb.get(self.adb_path,"r")
        else:
            return None

    def _gen_transformations_hash(self, count_key=True, key=None):
        """
        ::

            Checks transformation parameters and generates a unique transformations key
            for the database pointed to by self.adb_path
        """
        adb = self._get_adb()
        key_mapper = KeyMapper(adb, self.num_patterns)
        key = key_mapper.map_forward(dict(self.seq_params, **self.syn_params), count_key, key)
        return key

    def _gen_patterns_hash(self, count_key=True):
        """
        ::

            Checks transformation parameters and generates a unique ground-truth key
            for the database pointed to by self.adb_path
        """
        adb = self._get_adb()
        key = HASH_KEY
        key_mapper = KeyMapper(adb, self.num_patterns)
        trans_key = key_mapper.map_forward(dict(self.seq_params, **self.syn_params))
        key = key_mapper.map_forward(trans_key, count_key, key=key)
        return key

    def synthesize_audio(self, patterns=None, syn_params=None, suffix=""):
        """
        ::

            Synthesize audio from self.patterns n-tuples and self.syn_params
            Write audio files to self.collection_path

            Arguments:
              patterns -  music sequences to synthesize in format expected by self.syn_fun
              suffix   -  unique identifier suffix if required [usually a parameter key]
        """
        if patterns==None:
            print "_synthesize_patterns: patterns must be specified"
            raise TestCollectionError()
        if syn_params==None:
            syn_params = self.syn_params
        print "Generating audio..."
        for i in range(len(patterns)):
            wav_name = self.collection_path + os.sep + "%06d"%i + suffix + self.uid + ".wav" 
            f = None
            try:
                f = open(wav_name, 'r')                
                continue
            except:
                pass
            finally:
                if f:
                    f.close()
            if type(patterns[0][0])==tuple:
                sig=[]
                for j in range(len(patterns[0][0])):
                    sig.append(self.syn_fun(syn_params, self.seq_params, patterns[i][j]))
            else:
                sig = self.syn_fun(syn_params, self.seq_params, patterns[i])
            if type(patterns[0][0])==tuple:
                signal = pylab.hstack(sig)
            else:
                signal = sig
            sound.wav_write(signal, wav_name, syn_params['sr'])
        return True

    def save(self):
        """
        ::

            Save current patterns and transformations, and data paths.
               self.collection_path - the location of the current data set
               self.collection_stem - the stem name of the collection
               self.adb_path - the name of the base rhythm patterns database
               self.num_patterns - number of rhythm patterns == len(self.adb.liszt())
               self.seq_params - pattern sequencer parameters
               self.syn_params - audio synthesis parameters
               self.feature_params - audio feature extraction parameters

               self.patterns - integer representation of each rhythm pattern
               self.transformations - integer representation of each rhythm transformation for n_flip_bits and [rand_channels]

        """
        f = open(self.collection_path + os.sep + self.collection_stem + self._gen_adb_hash() + self.uid + ".data","w")
        pickle.dump((self.collection_path, 
                     self.collection_stem, 
                     self.uid,
                     self.adb_path, 
                     self.num_patterns, 
                     self.seq_params, 
                     self.syn_params, 
                     self.feature_params), f)
        f.close()
        
    def load(self, data_path=None):
        """
        ::

            Load patterns and data paths:
               self.collection_path - the location of the current data set
               self.collection_stem - the stem name of the collection
               self.patterns - integer representation of each rhythm pattern
               self.adb_path - the name of the base rhythm patterns database
               self.num_patterns - number of rhythm patterns == len(self.adb.liszt())
               self.seq_params - pattern sequencer parameters
               self.syn_params - audio synthesis parameters
               self.feature_params - audio feature extraction parameters
        """
        if data_path==None:
            print "You must supply a data_path to load from."
            return
        if data_path[-4:] != "data":
            data_path = self.toc(data_path)[0][0]
        f = open(data_path,"r")
        l = pickle.load(f)
        f.close()
        self.collection_path = l[0]
        self.collection_stem = l[1]
        self.uid = l[2]        
        self.adb_path = l[3]
        self.num_patterns = l[4]
        self.seq_params = l[5]
        self.syn_params = l[6]
        self.feature_params = l[7]
        self._load_gt_patterns()

    def _remove_temporary_files(self, key="", features_only=False):
        """
        ::

            Remove temporary synthesized audio files and feature and power files.
            Only remove files after sequencer, synthesis, analysis, and insertion operations
            Caution: removing files out of turn can break stuff
        """
        audiocollection.AudioCollection._remove_temporary_files(self, key + self.uid)
        if not features_only:
            aList = glob.glob(self.collection_path + os.sep + "*" + key + self.uid + ".wav")
            for a in aList: os.remove(a) # remove synthetic audio data

    def setup_evaluation(self, test_set_list):
        """
        ::

            Load a adb database as a python object
            Initialize search parameters
            Evaluate ranks and distances for each ground-truth pattern against a set of transformations
            Level-1 evaluation:
               - retrieval of ground-truth patterns from a collection of transformed patterns
        """
        adb = audiodb.adb.get(self.adb_path,"r")
        lzt = adb.liszt()
        keys,lens = zip(*lzt)
        km = KeyMapper(adb, self.num_patterns)
        klist = km.list_key_instances()
        if not len(klist):
            return None, None
        includeKeys=[]
        while len(test_set_list):
            test_set_pos = klist[test_set_list.pop()][1]
            test_set = keys[test_set_pos+self.num_patterns:test_set_pos+2*self.num_patterns]
            includeKeys.extend(test_set)
        adb.configQuery['absThres']=-6.0
        adb.configQuery['accumulation']='db'
        adb.configQuery['npoints']=len(lzt) # All points in database
        adb.configQuery['ntracks']=0
        adb.configQuery['distance']='euclidean'
        adb.configQuery['radius']=0.0
        adb.configQuery['seqLength']=lzt[0][1]
        adb.configQuery['seqStart']=0
        adb.configQuery['hopSize']=1
        adb.configQuery['includeKeys']=includeKeys
        return adb, lzt

    def _initialize_test_set_lists(self, test_set_key, clear_lists):
        """
        ::

            Returns test_sets and test_set_lists for current audioDB
            for given arguments:
             test_set_key: hash key of a transformation set
             clear_lists: set to True to reset ranks_list and dists_list, otherwise
             these will be appended with next call to evaluate()
        """
        test_sets = self.get_test_sets()
        if not len(test_sets):
            print "Empty audioDB instance: ", self.adb_path
            return False
        if test_set_key == None:
            test_set_list = range(len(test_sets))
        else:
            print "Evaluating:", test_set_key, "..."
            # find test_set corresponding to key
            keys, counts = zip(*test_sets)
            try:
                key_idx = keys.index(test_set_key)
                test_set_list = [key_idx]
            except ValueError:
                print "Key not found in audioDB test_sets: ", test_set_key
                raise TestCollectionError()                        
        self.test_set_list=test_set_list
        if clear_lists or self.ranks_list==None:
            self.ranks_list=[]
            self.dists_list=[]
        return test_sets, test_set_list
            
    def evaluate(self, test_set_key=None, clear_lists=True):
        """
        ::

            Evaluate ranks and distances for each ground-truth pattern against a set of rhythm transformations
            by retrieval of transformed patterns using original patterns as queries.

            test_set_key: a test-set hash-key identifier as returned by get_test_lists()
            clear_lists: set to True to make a fresh set of results lists for this evaluation
                         set to False to append current evaluation to previsouly stored lists
            Returns a tuple (ranks,dists)
            Each element of the tuple contains len(test_set_list) lists, 
             each list contains ground-truth ranks, and whole-dataset distances, of each test-set 
             in the test_set_list
        """
        test_sets, test_set_list = self._initialize_test_set_lists(test_set_key, clear_lists)
        for test_set in test_set_list:
            print "Evaluating test set:", test_set
            ranks=[]
            dists=[]
            adb, lzt = self.setup_evaluation([test_set])
            if not len(lzt):
                print "Test-set not found in audioDB: ", test_set
                raise TestCollectionError()
            qkeys, qlens = zip(*lzt)
            start_pos = test_sets[ test_set ][ 1 ]
            end_pos = start_pos + self.num_patterns
            for q,qkey in enumerate( qkeys[ start_pos : end_pos ] ):
                res = adb.query(key=qkey).rawData
                if len(res):
                    rkeys, dst, qpos, rpos = zip(*res)
                    dists.extend(dst) # All result distances
                    try: 
                        idx = rkeys.index(adb.configQuery['includeKeys'][q]) # key match
                        ranks.append(idx) # Rank of the transformation
                    except:
                        print "Key not found in result list: ", qkey
                        sys.stdout.flush()
                        ranks.append(-1)
                else:
                    print "Empty result list: ", qkey
                    sys.stdout.flush()
                    ranks.append(-1)
            self.ranks_list.append((test_set,ranks))
            self.dists_list.append((test_set,dists))
        self.print_results()
        return True

    def print_results(self, test_set_list=None):
        """
        ::

            Display the mean ranks of rank lists.
            """
        if self.ranks_list == None:
            print "print_results requires self.ranks_list to be computed or loaded"
            return False
        if test_set_list == None:
            test_set_list = self.test_set_list
        mean_ranks = self.get_mean_ranks()
        klist = self.get_test_sets()
        if not len(klist):
            print "Skipping empty audioDB: ", self.adb_path
            return False
        a = self.key_map_inverse(klist[0][0])
        print "Index %s MeanRank" %a.keys()
        width = len(a.values().__repr__()) + 6
        print '=' * width
        for test_set in mean_ranks.keys():
            a = self.key_map_inverse(klist[test_set][0])
            strval = a.values()
            print "[%3d] %s %f" %(test_set, strval, mean_ranks[test_set])
        print '=' * width
        return True
    
    def get_mean_ranks(self):
        """
        ::

            Display mean ranks for each test set
            Requires either load_results or evaluate() to have been called
        """

        if self.ranks_list:
            mr=dict()
            for i,r in self.ranks_list:
                mr[i]=pylab.mean(r)
            return mr
        else:
            print "get_mean_ranks requires either load_results or evaluate() to have been called"
            return None
    
    def save_results(self):
        """
        ::

            Persist last evaluation result set to a file
        """
        if not (self.test_set_list and self.ranks_list and self.dists_list):
            print "save_results requires either load_results or evaluate() to have been called"
            return False
        self.test_set_list = range(len(self.get_test_sets()))
        f = open(self.collection_path + os.sep + self.collection_stem + self._gen_adb_hash() + self.uid + ".results","w")
        pickle.dump((self.test_set_list, self.ranks_list, self.dists_list), f)
        f.close()
        return True

    def load_results(self):
        """
        ::

            Load a previously-saved evaluation result set
        """
        if not self.collection_path:
            print "load_results requires previous calls to evaluate() and save_results()"
            return False
        f=None
        fname = self.collection_path + os.sep + self.collection_stem + self._gen_adb_hash() + self.uid + ".results"
        try:
            f = open(fname,"r")
        except:
            print "Cannot open --->", fname
            return False
        self.test_set_list, self.ranks_list, self.dists_list = pickle.load(f)
        f.close()
        self.test_set_list = range(len(self.get_test_sets()))
        return True

    def get_test_sets(self):
        """
        ::

            Return a list of data sets in this database instance.
            Each data set is indexed by the transformation parameters using a hash code.
            Mutliple instances of the same parameters are allowed and are counted.
            Use print_test_sets() to get a more readable display
        """
        adb = audiodb.adb.get(self.adb_path,"r")
        km = KeyMapper(adb, self.num_patterns)
        return km.list_key_instances()

    def key_map_inverse(self, key):
        """
        ::

            Return parameter dict for give key
        """
        adb = audiodb.adb.get(self.adb_path,"r")
        km = KeyMapper(adb, self.num_patterns)
        return km.map_inverse(key)


    def print_test_sets(self):
        """
        ::

            Print the test sets in a readable format:
        """
        klist = self.get_test_sets()
        adb = audiodb.adb.get(self.adb_path,"r")
        km = KeyMapper(adb, self.num_patterns)
        print "Index"
        print '-'*45
        for i,n in enumerate(klist):
            a = km.map_inverse(n[0])
            print "[%d] %s" %(i,a)
        print '-'*45
        return True

    # Evaluation loop for TestCollection objects
    def evaluation_loop(self, sequencer_dict_list=None, synthesizer_dict_list=None, analyzer_dict_list=None):
        """    
        ::

            Comprehensive evaluation: a time and space efficient workflow
            Iterations over sequencing, synthesis, and analysis parameters
        """
        if not (self.collection_path):
            print "evaluation_loop requires either self.initialize() or self.load(path...) first"
            return False
        sequencer_list = self._dict_list_to_tuples(sequencer_dict_list, self.seq_params)
        synthesizer_list = self._dict_list_to_tuples(synthesizer_dict_list, self.syn_params)
        analyzer_list = self._dict_list_to_tuples(analyzer_dict_list, self.feature_params)        
        print sequencer_list
        print synthesizer_list
        print analyzer_list
        self.cache_temporary_files=False # Automatically remove temporary files
        self.ranks_list = []
        self.dists_list = []
        # Main parameter iteration loop
        for s_tuple in synthesizer_list:
            for s_key, s_val in s_tuple:
                self.syn_params[s_key] = s_val # change one synthesis parameter
                print s_key, s_val
            gt_key=self._gen_patterns_hash(count_key=False)
            self.synthesize_audio(patterns=self.patterns, suffix=gt_key)
            for p_tuple in sequencer_list:
                for p_key, p_val in p_tuple:                    
                    self.seq_params[p_key] = p_val # change one transformation parameter
                    print p_key, p_val
                self.gen_transformation_set(save_transformations=True) # Generate controlled random transformations
                trans_key = self._gen_transformations_hash(count_key=False)
                self.synthesize_audio(patterns=self.transformations, 
                                      syn_params=self.seq_params['syn_params'], suffix=trans_key)
                for f_tuple in analyzer_list:
                    for f_key, f_val in f_tuple:
                        self.feature_params[f_key] = f_val # change one feature parameter
                        print f_key, f_val
                    print
                    curr_gt_key, curr_trans_key = self.extract_features() 
                    self.load_results()
                    self.evaluate(curr_trans_key, clear_lists=False) # output results
                    self.save_results()
                self._remove_temporary_files(key=trans_key)
            self._remove_temporary_files(key=gt_key)
        return True

    @staticmethod
    def _print_dict_list_tuples(dict_list=None, params_dict=None):
        """
        ::

            Inspect a dict_tuple generated from a dict_list
        """
        s = TestCollection._dict_list_to_tuples(dict_list, params_dict)
        for s_tuple in s:
            for s_key, s_val in s_tuple:
                print s_key, s_val
            print
        
    @staticmethod
    def _dict_list_to_tuples(dict_list=None, params_dict=None):
        """
        ::

            Map from parameter dicts with list values to iterated parameter tuples:
              raw dict for serial iteration:
                   {'a': [0,1,2], 'b': [2,3,4]} -> (('a',0),('a',1),('a',2),('b',2),('b',3),('b',4))
              tuple of dict to co-vary parameters:
                   ({'a': [0,1,2], 'b': [2,3,4]},) -> (('a',0,'b',2),('a',1,'b',3),('a',2,'b',4))
              list of tuple of dict to nest:
                   (({},...),({},...),...) -> (('a',0,'b',2),('a',1,'b',3),('a',2,'b',4), ... )
            Syntax and Semantics:
             {k:[v1,v2,...,vm],l:[u1,u2],...,um}: iterate value lists serially: (k,v1),(k,v2),...,(l,u1),...,(l,um)
             (k:[v1,v2,...,vm],l:[u1,u2],...,um},): iterate value lists by dot product: ((k,v1),(l,u1)),((k,v2),(l,v2)),...
             ({},{},{},...): dot-product of dicts
             (({},...),({},...),...): serial iteration of dot products of dicts
        """
        out_list = []
        if dict_list == None:
            if params_dict == None:
                print "You must specify a params_dict as default value for dict_list"
                raise TestCollectionError()
            else:
                out_list.append([params_dict.iteritems().next()])
        else:
            if type(dict_list) == tuple or type(dict_list) == list:
                if len(dict_list)>0 and type(dict_list[0]) == tuple or type(dict_list[0])==list:
                    for d in dict_list: out_list.extend(TestCollection._dict_list_to_tuples(d)) 
                else:
                    d_list = []
                    for d in dict_list:
                        for k in d.keys():
                            d_list.append([t for t in zip([k]*len(d.get(k)),d.get(k))])
                    tuple_list = zip(*d_list)
                    out_list.extend(tuple_list)
            elif type(dict_list) == dict:
                for k in dict_list.keys(): 
                    tuple_list = [(t,) for t in zip([k]*len(dict_list.get(k)),dict_list.get(k))]
                    out_list.extend(tuple_list)
            else:
                print "Error in _dict_list_to_tuples. Parameter specification is not dict or list of dicts."
                raise TestCollectionError()
        return out_list

    
    def extract_features(self, extract_only=False, wave_suffix=".wav"):
        """
        ::

            Extract features from audio collection and insert into audiodb instance in the following order:
               check for an audioDB instance on current feature_params, create if none
               ground truth patterns - hex32hash(0) + hex4(key_count) -> audioDB
               variation patterns    - hex32hash(seq_params + self.syn_params) + hex4(key_count) -> audioDB
               insert_features_into_audioDB -> audiodb

               an instance is a test_set instance consisting of self.num_patterns audio files.
        """
        print "Extracting features / audioDB insertion..."
        self._new_adb() 
        ext = self.uid + wave_suffix

        repl_gt_key=self._gen_patterns_hash(count_key=False)
        gt_key = self._gen_patterns_hash()
        self.insert_audio_files(glob.glob(self.collection_path + os.sep + "*" + repl_gt_key + ext))
        audiocollection.AudioCollection.extract_features(self, key=gt_key+ext, keyrepl=repl_gt_key+ext, extract_only=extract_only)

        repl_trans_key = self._gen_transformations_hash(count_key=False)
        trans_key = self._gen_transformations_hash()
        self.insert_audio_files(glob.glob(self.collection_path + os.sep + "*" + repl_trans_key + ext))
        audiocollection.AudioCollection.extract_features(self, key=trans_key+ext, keyrepl=repl_trans_key+ext, extract_only=extract_only)

        self._remove_temporary_files(features_only=True)
        self.save()
        return (gt_key, trans_key)

    def collection_results(self, plotting=False, title_param=None, evaluate=False):
        """
        ::

            Collect, print, and, optionally, reevaluate all the results in the collection.
            Re-evaluation collects all the results into the current instance, which are then saved.
        """
        if not self.collection_path:
            print "collection_results requires self.collection_path to be set."
            return False
        dList = self.toc()
        if plotting: pylab.figure()
        sub=1
        for d,t in dList:
            print "*** Evaluating AudioDB Instance:", d, "***"
            self.load(d)
            print self.feature_params.keys()
            print self.feature_params.values()
            if evaluate:
                self.evaluate()
                self.save_results()
            else:
                if not self.load_results():
                    print "No saved results, skipping..."
                    continue
                if not self.print_results():
                    print "Error printing results..."
                    continue
            if plotting:
                pylab.subplot(int(round(len(dList)/2.0)),2,sub)
                mr = self.get_mean_ranks()
                pylab.stem(mr.keys(), mr.values())
                if title_param:
                    pylab.title(title_param + " = {0}".format(self.feature_params[title_param]))
                sub+=1
        return True

class KeyMapper:
    """
    ::

        A helper class to manage key encodings for TestCollection instances
    """
    def __init__(self, adb=None, num_patterns=100):
        """
        Initialize with an adb instance
        """
        self.adb = adb
        self.num_patterns = num_patterns
        if adb:
            self.collection_path = self.adb.path.rpartition(os.sep)[0]

    def map_forward(self, param_dict, count_key=True, key=None, suffix=""):
        """
        ::

            param_dict -> key="md5 hash on sorted dict"+".instance_count"
        """
        if not key:
            key = self._gen_hash_key(param_dict)
        if count_key and self.adb: 
            c = self._count_key_instances(key)        
            key = key+"%04X"%c # length-36 key = MD5 parameter hash + 4 count nibbles
            dname = self.collection_path + os.sep + ".key" + key + suffix
            if not self._cache_key_params(dname, param_dict):
                print "Error saving key cache"
                raise TestCollectionError()
        else:
            key = key+"%04X"%0 # length-36 key = MD5 parameter hash + 4 count nibbles        
        return key

    def _gen_hash_key(self, param_dict):
        m = hashlib.md5()
        keys = param_dict.keys()
        keys.sort() # ensure vals ordered by keys in lexicographical order
        vals = [param_dict.get(k) for k in keys]
        m.update(vals.__repr__())
        key = m.hexdigest()
        return key

    def _cache_key_params(self, dname, param_dict):
        f = None
        save_dict = False
        try:
            f = open(dname,"r")
        except:
            save_dict = True
        finally:
            if f:
                f.close()
        if save_dict:
            try:
                f = open(dname,"w")
                pickle.dump(param_dict, f)
            except:
                print "Error opening key dict cache file: ", dname
                return False
            finally:
                if f:
                    f.close()
        return True

    def _count_key_instances(self, key):
        """
        ::

            adb:key="md5 hash on sorted values" -> count
        """
        if not self.adb:
            return 0
        lzt = self.adb.liszt()
        if not len(lzt):
            return 0
        keys,lens = zip(*lzt)
        count=0
        that_key = BASE_KEY+"%s"%key
        that_key = that_key[0:len(BASE_KEY)+len(HASH_KEY)] # remove instance count
        for k in keys[0:-1:self.num_patterns]:
            pth, sep, this_key = k.rpartition(os.sep)            
            if this_key.startswith(that_key):
                count+=1
        return count

    def list_key_instances(self):
        """
        ::

            Search adb.liszt() for instances of GROUND_TRUTH_KEY, list (KEY,POS)
        """
        lzt = self.adb.liszt()
        if not len(lzt):
            return tuple([])
        keys,lens = zip(*lzt)
        key_list=[]
        gt_key = "%032X"%0
        for i,k in enumerate(keys):
            pth, sep, this_key = k.rpartition(os.sep)
            if this_key.startswith(BASE_KEY+gt_key):
                trans_key = self.map_inverse(this_key[len(BASE_KEY):len(BASE_KEY+HASH_KEY+COUNT_KEY)])
                key_list.append((trans_key,i)) # return trans_key for this gt_key
        return tuple(key_list)

    def map_inverse(self, key, suffix=""):
        """
        ::

            key="md5 hash on sorted values" -> dict(keys, values)
        """
        dname = self.collection_path + os.sep + ".key" + key + suffix
        f = None
        d = None
        try:
            f = open(dname,"r")
            d = pickle.load(f)
        except:
            print "Key not in collection: ", key
        finally:
            if f:
                f.close()
        return d

class RhythmTest(TestCollection):
    """
    ::

        A TestCollection class for rhythm sequences

        Input arguments;
          Stored instance:
           path: directory containing an instance of a previously saved RhythmTest [None]
                 Data will be loaded into this instance ready for adding more test sets and evaluating

          New instance:
           num_patterns: how many rhythm patterns to generate in this collection [100]
           root_path: path to root directory for holding rhythm test collections [audiocollection.DATA_ROOT="$HOME/exp"]

          Data members:
            patterns  - integer bit-field representation of num_patterns patterns for subdiv subdivisions
            transformations - random transformations of each pattern by flipping n_flip_bits in each of rand_channels
            seq_params - pattern generation and transformations parameters
            syn_params  - audio synthesis from patterns parameters
            syn_fun - audio synthesis algorithm [testsignal.rhythm]
            feature_params - audio feature extraction parameters

            default syn_params dict for the collection:
                'sr' : 48000,        # sample rate
                'bw' : [80., 2500., 1000.], # band-widths
                'cf' : [110., 5000., 16000.], # center-frequencies
                'dur': [0.5, 0.5, 0.5] # relative duration of timbre
                'normalization' : 'none' # balance timbre channels 'none', 'maxabs', 'rms', 'norm'

            default seq_params dict for the collection:
                'poiss_channels' : [.25, .25, .25], # Poisson parameter per channel
                'n_flip_bits' : 1, # number of perturbed (transformed) subdivisions
                'rand_channels' : [0], # timbre channels on which to perform n_flip_bits
                'subdiv' : 16, # subdivision of 4/4 measure
                'tempo' : 120. # tempo of rhythmic sequences
    """
    collection_stem = "rhythmtest_"
    def __init__(self, path=None, num_patterns=100, root_path=audiocollection.DATA_ROOT):
        load_path=None
        if path!=None and (path[-5:]=='.data' or path[-4:]=='.adb'):
            load_path=path
            path=path.rpartition(os.sep)[0]
        TestCollection.__init__(self, num_patterns, path, root_path)
        self._set_default_params()
        if load_path!=None:
            self.load(load_path)

    def _set_default_params(self):
        """
        Sets default params as described in class header
        """
        self.syn_fun=testsignal.rhythm # the audio synthesis function for this TestCollection
        self.syn_params, self.seq_params, patterns = testsignal.default_rhythm_params()
        self.syn_params['dur'] = [0.5, 0.5, 0.5]

        self.seq_params['poiss_channels'] = [.25, .25, .25]
        self.seq_params['n_flip_bits'] = 1
        self.seq_params['rand_channels'] = [0]
        self.seq_params['delta_cf'] = [0.0, 0.0, 0.0] # arithmetic transformation deltas
        self.seq_params['delta_bw'] = [0.0, 0.0, 0.0]
        self.seq_params['delta_dur'] = [0.0, 0.0, 0.0]

        self.feature_params['feature']='cqft'
        self.feature_params['ncoef']=1
        self.feature_params['log10']=False
        self.feature_params['magnitude']=True
        self.feature_params['sample_rate']=48000
        self.feature_params['nfft']=4096
        self.feature_params['wfft']=4096
        self.feature_params['nhop']=3000
        
    def run_test(self):
        """
        ::

            Pre-conditions:
              self.intialize() has been called
              self.syn_params is set to desired values
              self.feature_params is set to desired values

            Post-conditions:
              Generated random synthetic rhythm patterns, synthesized audio, extracted features, persisted in an adb database
              Generated pattern transformations, synthesized audio, extracted features, persisted in an adb database
              Saved state and data in an audioDB instance identified by feature_params
                the corresponding adb_database name is unique to self.feature_params.

        """
        if not self.patterns:
            print "You must call either initialize() or load() before run_test()"
            return False
        key = self._gen_patterns_hash(count_key=False)
        self.synthesize_audio(patterns=self.patterns, suffix=key)
        self.gen_transformation_set(save_transformations=True)
        key = self._gen_transformations_hash(count_key=False)
        self.synthesize_audio(patterns=self.transformations, suffix=key)
        curr_gt, curr_trans = self.extract_features()
        self.save()
        self.evaluate(curr_trans)
        self.save_results()
        self._remove_temporary_files()
        return True

    def _check_syn_params(self, patterns=None, syn_params=None):
        if patterns==None:
            patterns=self.patterns
        if syn_params==None:
            syn_params = self.syn_params
        num_timbres = len(syn_params['cf'])
        if not ( num_timbres == len(syn_params['bw']) == len(syn_params['dur']) == len(patterns) ):
                print "Timbre-channel mismatsh between syn_params, seq_params, and num_timbre_channels "
                raise TestCollectionError()
        return num_timbres

    def _check_seq_params(self):
        if self.seq_params['n_flip_bits']==None:
            print "Must have defined self.n_flip_bits as int"
            return False

        if self.seq_params['rand_channels']==None or type(self.seq_params['rand_channels'])!=list:
            print "Must have defined self.rand_channels as a list"
            return False
        
        if not (len(self.seq_params['delta_cf']) 
                == len(self.seq_params['delta_bw']) 
                == len(self.seq_params['delta_dur']) 
                == len(self.syn_params['cf'])
                == len(self.syn_params['bw'])
                == len(self.syn_params['dur'])):
            print "Inconsistent delta_cf/cf, delta_bw/bw, delta_dur/dur"
            return False        
        return True

    def transform_patterns(self):
        """
        ::

            Given a set of M base patterns over T timbres perform
            transformation by:
               n_flip_bits: randomly flipping n pattern bits in rand_channels
               delta_cf: add amount to cf of each channel
               delta_bw: add amount to bw of each channel
        """
        if not self._check_seq_params():
            return False
        self.transformations=[]
        for p in self.patterns:
            mut = []
            for c in range(len(p)):
                if self.seq_params['rand_channels'].count(c):
                    mut.append(self._transform_pattern(p[c]))
                else:
                    mut.append(p[c])
            self.transformations.append(mut)
        self.seq_params['syn_params'] = self.syn_params.copy()
        self.seq_params['syn_params']['cf'] = [a+b for a,b in zip(self.syn_params['cf'], self.seq_params['delta_cf'])]
        self.seq_params['syn_params']['bw'] = [a+b for a,b in zip(self.syn_params['bw'], self.seq_params['delta_bw'])]
        self.seq_params['syn_params']['dur'] = [a+b for a,b in zip(self.syn_params['dur'], self.seq_params['delta_dur'])]
        return True
    
    def _transform_pattern(self, pat):
        """
        ::

            flip random n-bits (n guaranteed) of bit pattern in integer pat
        """
        i_list = []
        i = pylab.randint(self.seq_params['subdiv'])
        for b in range(self.seq_params['n_flip_bits']):
            while(i_list.count(i)):
                i = pylab.randint(self.seq_params['subdiv'])
            pat ^= (1 << i)
            i_list.append(i)
        return pat

    def sequence_patterns(self):
        """
        ::

            Map: self.seq_params -> num_patterns x random sequence
            Generate a set of poisson random percussive patterns from self.seq_params
            Returns a list of multi-timbre patterns as (a,b,c,...) integer n-tuples
        """
        patterns = []        
        for i in pylab.arange(self.num_patterns):
            rand_sequences = []
            for pc_mu in self.seq_params['poiss_channels']:
                rand_sequences.append(scipy.stats.poisson.rvs(pc_mu, 0, size=self.seq_params['subdiv']))
            r_all = zip(*rand_sequences)
            bit_patterns = [0]*len(rand_sequences)
            for j,z in enumerate(r_all):
                for k in range(len(bit_patterns)):
                    bit_patterns[k] |= (z[k]>0) << (self.seq_params['subdiv']-j-1)
            patterns.append(bit_patterns)
        return patterns

    def _dec_to_bin(self,pat):
        """
        ::

            Convert rhythm pattern to binary string representation
            Oeprates on single pattern self.pattern[0][0] or tuple of patterns self.pattern[0]
            Returns string or tuple of strings representing rhythm patterns for self.seq_params['subdiv'] subdivisions
        """
        if type(pat)==list or type(pat)==tuple:
            s=[]
            for p in pat:
                s.append(pylab.binary_repr(p,width=self.seq_params['subdiv']))
        else:
            s = pylab.binary_repr(pat,width=self.seq_params['subdiv'])
        return s

    @staticmethod
    def _hamming_distance(pat1, pat2):
        """
        ::

            Returns the hamming distance between two Integer bit patterns
        """
        if type(pat1)==list or type(pat1)==tuple:
            s=[]
            for i,p1 in enumerate(pat1):
                s.append([abs(((p1&1<<k)>>k) - ((pat2[i]&1<<k)>>k)) for k in range(32)])
        else:
            s = [abs(((pat1&1<<k)>>k) - ((pat2&1<<k)>>k)) for k in range(32)]
        return pylab.sum(s)

# RHYTHM STREAMS
class RhythmStreamTest(RhythmTest):
    """
    ""

        A TimbreCollection class based on RhythmTest for rhythm sequences in timbre channels

        Input arguments;
          Stored instance:
           path: directory containing an instance of a previously saved RhythmTest [None]
                 Data will be loaded into this instance ready for adding more test sets and evaluating

          New instance:
           num_channels: how many timbre channels [3]
           num_patterns: how many rhythm patterns to generate in this collection [100]
           root_path: path to root directory for holding rhythm test collections [audiocollection.DATA_ROOT="$HOME/exp"]

          Data members:
            patterns  - integer bit-field representation of num_patterns patterns for subdiv subdivisions
            transformations - random transformations of each pattern by flipping n_flip_bits in each of rand_channels
            seq_params - pattern generation and transformations parameters
            syn_params  - audio synthesis from patterns parameters
            syn_fun - audio synthesis algorithm [testsignal.rhythm]
            feature_params - audio feature extraction parameters

            default syn_params dict for the collection:
                'sr' : 48000,        # sample rate
                'bw' : [80., 2500., 1000.], # band-widths
                'cf' : [110., 5000., 16000.], # center-frequencies
                'dur': [0.5, 0.5, 0.5] # relative duration of timbre
                'normalization' : 'none' # balance timbre channels 'none', 'maxabs', 'rms', 'norm'

            default seq_params dict for the collection:
                'poiss_channels' : [.25, .25, .25], # Poisson parameter per channel
                'n_flip_bits' : 1, # number of perturbed (transformed) subdivisions
                'rand_channels' : [0], # timbre channels on which to perform n_flip_bits
                'subdiv' : 16, # subdivision of 4/4 measure
                'tempo' : 120. # tempo of rhythmic sequences
    """

    collection_stem = "rhythmstreamtest_"
    def __init__(self, path=None, num_channels=3, num_patterns=100, root_path=audiocollection.DATA_ROOT):
        RhythmTest.__init__(self, path, num_patterns, root_path)
        self.syn_params['num_timbre_channels'] = num_channels
        self.feature_params['num_timbre_channels'] = num_channels

    def synthesize_audio(self, patterns=None, syn_params=None, suffix=''):
        """
        ::

            Overridden method from RhythmTest for timbre channels
            Synthesize audio into separate timbre-channel wave files

            Create wavs from self.patterns n-tuples and self.syn_params
               Write audio files to self.collection_path

               Arguments:
                 patterns -  music sequences to synthesize in format expected by self.syn_fun            
                 suffix   -  unique identifier suffix if required [usually a parameter key]

               Special case: syn_params['num_timbre_channels'] must be same value as feature_params[...],
               required for rendering pattern audio into separate timbre channels prior to feature extraction
        """
        if patterns==None or not len(patterns):
            print "_synthesize_patterns: patterns must be specified"
            raise TestCollectionError()        
        if syn_params==None:
            syn_params=self.syn_params
        print "Generating timbre-channel audio..."
        num_timbres = self._check_syn_params(patterns[0], syn_params)
        if num_timbres != syn_params['num_timbre_channels']:
            print "num_timbre_channels mismatch in syn_params"
            raise TestCollectionError()
        for i in range(len(patterns)):
            for t_chan in range(syn_params['num_timbre_channels']):
                wav_name = self.collection_path + os.sep + "%03d"%i + "%03d"%t_chan + suffix + self.uid + ".wav" 
                f = None
                try:
                    f = open(wav_name, 'r')
                    continue
                except:
                    pass
                finally:
                    if f:
                        f.close()
                synth_params = syn_params.copy()
                seq_params = self.seq_params.copy()
                synth_params['cf']=[self.syn_params['cf'][t_chan]]
                synth_params['bw']=[self.syn_params['bw'][t_chan]]
                synth_params['dur']=[self.syn_params['dur'][t_chan]]
                if type(patterns[i][0])==tuple:
                    sig=[]
                    for j in range(len(patterns[0][0])):
                        sig.append(self.syn_fun(synth_params, seq_params, (patterns[i][j][t_chan],)))
                    sig = pylab.hstack(sig)
                else:
                    sig = self.syn_fun(synth_params, seq_params, (patterns[i][t_chan],))
                wav_name = self.collection_path + os.sep + "%03d"%i + "%03d"%t_chan + suffix + self.uid + ".wav" 
                sound.wav_write(sig, wav_name, syn_params['sr'])
        return True
        
    def setup_evaluation(self, test_set_list, timbre_channel_offset):
        """
        ::

            Overridden method for timbre-channel setup evaluation.
            Load a adb database as a python object
            Initialize search parameters
            Evaluate retrieval of each ground-truth pattern against a set of transformations
            Level-1 evaluation:
               - retrieval of ground-truth patterns from a collection of transformed patterns
        """
        adb = audiodb.adb.get(self.adb_path,"r")
        lzt = adb.liszt()
        if not len(lzt):
            print "AudioDB empty in setup_evaluation: ", self.adb_path
            raise TestCollectionError()
        keys,lens = zip(*lzt)
        km = KeyMapper(adb, self.num_patterns)
        klist = km.list_key_instances()
        if not len(klist):
            return None, None
        includeKeys=[]
        while len( test_set_list ):
            test_set_pos = klist[ test_set_list.pop() ][ 1 ]
            start_pos = test_set_pos + self.num_patterns * self.feature_params['num_timbre_channels']
            end_pos = start_pos + self.num_patterns * self.feature_params['num_timbre_channels']
            test_set = keys[ start_pos + timbre_channel_offset : end_pos : self.feature_params['num_timbre_channels'] ]
            includeKeys.extend(test_set)
        adb.configQuery['absThres']=-6.0
        adb.configQuery['accumulation']='db'
        adb.configQuery['npoints']=len(lzt) # All points in database
        adb.configQuery['ntracks']=0
        adb.configQuery['distance']='euclidean'
        adb.configQuery['radius']=0.0
        adb.configQuery['seqLength']=lzt[0][1]
        adb.configQuery['seqStart']=0
        adb.configQuery['hopSize']=1
        adb.configQuery['includeKeys']=includeKeys
        return adb, lzt


    def evaluate(self, test_set_key=None, clear_lists=True):
        """
        ::

            Overridden method from RhythmTest for timbre channels
             Map distances and weights for each query in each channel
             Reduce using Bhattacharyya distance metric.

             Sets self.ranks_list, self.dists_list
             Display ranked results as in print_results()

            Evaluate ranks and distances for each ground-truth pattern against a set of rhythm transformations
            by retrieval of transformed patterns using original patterns as queries.

            test_set_key: a test-set hash-key identifier as returned by get_test_lists()
            clear_lists: set to True to make a fresh set of results lists for this evaluation
                         set to False to append current evaluation to previsouly stored lists
            Returns a tuple (ranks,dists)
            Each element of the tuple contains len(test_set_list) lists, 
             each list contains ground-truth ranks, and whole-dataset distances, of each test-set 
             in the test_set_list
        """
        test_sets, test_set_list = self._initialize_test_set_lists(test_set_key, clear_lists)
        for test_set in test_set_list:
            print "Evaluating test set:", test_set
            t_rkeys=[]
            t_ikeys=[]
            t_dists=[]
            start_pos = test_sets[ test_set ][ 1 ]
            end_pos = test_sets[ test_set ][ 1 ] + self.num_patterns * self.feature_params['num_timbre_channels']
            for t_chan in range( self.feature_params['num_timbre_channels'] ):
                adb, lzt = self.setup_evaluation([test_set], t_chan)
                t_ikeys.append(adb.configQuery['includeKeys']) # timbre-channel keys
                t_rkeys.append([]) # per-query result keys
                t_dists.append([]) # timbre-channel dists (re-combined later)
                qkeys, qlens = zip(*lzt)
                for q,qkey in enumerate( qkeys[ start_pos + t_chan : end_pos : self.feature_params['num_timbre_channels'] ] ):
                    res = adb.query(key=qkey).rawData
                    if len(res):
                        rkeys, dst, qpos, rpos = zip(*res)
                        t_dists[t_chan].append(dst) # All result distances
                        t_rkeys[t_chan].append(rkeys) # timbre-channel distance-sorted keys
                    else:
                        print "Empty result list: ", qkey
            ranks, dists = self.timbre_channel_distance(t_ikeys, t_rkeys, t_dists)
            self.ranks_list.append((test_set, ranks))
            self.dists_list.append((test_set, dists))            
        self.print_results()
        return True


    def timbre_channel_distance(self, ikeys, rkeys, dists):
        """
        ::

            Reduce timbre-channel distances to ranked list by relevant-key-index 
        """
        # O(n^2) search using pre-computed distances
        ranks_list = []
        dists_list = []
        for r in range(self.num_patterns): # relevant key
            rdists=pylab.zeros(self.num_patterns)
            for i in range(self.num_patterns): # result keys
                for t_chan in range(self.feature_params['num_timbre_channels']): # timbre channels
                    try: 
                        # find dist for pattern i for query q
                        i_idx = rkeys[t_chan][r].index( ikeys[t_chan][i] ) # dataset key match
                        # the reduced distance function in include_keys order
                        # distance is the sum for now
                        rdists[i] += dists[t_chan][r][i_idx]
                    except:
                        print "Key not found in result list: ", ikeys[t_chan][i]
                        sys.stdout.flush()
            #search for the index of the transformation
            sort_idx = pylab.argsort(rdists)   # Sort fields into database order
            ranks_list.append(pylab.where(sort_idx==r)[0][0]) # Rank of the relevant key
            dists_list.append(list(rdists))
        return (ranks_list, dists_list)


# PLCA TIMBRE CHANNELS
class RhythmPLCATest(RhythmStreamTest):
    """
    ::

        A class to create Rhythm Streams in PLCA timbre channels. Works as for RhythmStreamTest but audio synthesis
        is RhythmTest (summed timbre channels) and extract_features performs timbre-channel extraction.    
    """
    collection_stem = "rhythmplcatest_"
    def __init__(self, path=None, num_components=3, num_channels=3, num_patterns=100, root_path=audiocollection.DATA_ROOT):
        RhythmStreamTest.__init__(self, path, num_channels, num_patterns, root_path)
        self.km = None # timbre-channel classifier
        self.syn_params['num_components'] = num_components
        self.feature_params['time_funs_only'] = True
        self.feature_params['power_ext'] = ".probs"

    def synthesize_audio(self, patterns=None, syn_params=None, suffix=''):
        """
        ::

            Generate audio from given rhythm patterns using self.syn_fun()
            Reverts to TestCollection.synthesize_audio because we want mixed audio to test PLCA extraction.
        """
        return TestCollection.synthesize_audio(self, patterns, syn_params, suffix)

    def extract_features(self, extract_only=False, wave_suffix=".wav"):
        """
        ::

            Override of TestCollection.feature_extraction() method to perform PLCA and KMeans cluster extraction
            Extract features from audio collection and insert into audiodb instance in the following order:
               Extract TestCollection features by feature_params[...]
               PLCA extraction and timbre-channel clustering on GT features (training)
               PLCA extraction and timbre-channel assignment on Transformation faetures (testing)
               Write new timbre-channel features (replacing old)
               insert_features_into_audioDB -> audiodb
               A TestCollection instance is a test_set instance consisting of self.num_patterns ground-truth patterns 
               audio files and self.num_patterns transformed patterns audio files.
        """

        print "Extracting features / audioDB insertion..."
        self._new_adb() 
        ext = self.uid + wave_suffix

        print "Extracting gt features / timbre channels..."                
        repl_gt_key=self._gen_patterns_hash(count_key=False)
        gt_key = self._gen_patterns_hash()
        self.insert_audio_files(glob.glob(self.collection_path + os.sep + "*" + repl_gt_key + ext))
        audiocollection.AudioCollection.extract_features(self, key=gt_key+ext, keyrepl=repl_gt_key+ext, extract_only=True)
        self.extract_timbre_channels(gt_key, repl_gt_key)
        self._insert_features_into_audioDB()

        print "Extracting transformations features / timbre channels..."
        repl_trans_key = self._gen_transformations_hash(count_key=False)
        trans_key = self._gen_transformations_hash()
        self.insert_audio_files(glob.glob(self.collection_path + os.sep + "*" + repl_trans_key + ext))
        audiocollection.AudioCollection.extract_features(self, key=trans_key+ext, keyrepl=repl_trans_key+ext, extract_only=True)
        self.extract_timbre_channels(trans_key, repl_trans_key, train=False)
        self._insert_features_into_audioDB()

        self._remove_temporary_files(features_only=True)
        self.save()
        return (gt_key, trans_key)

    def extract_timbre_channels(self, key, keyrepl, train=True, wave_suffix=".wav"):
        """
        ::

            For given audio files in test_collection:
              clear rTupleList (list of keys and feature/power files to insert against associated keys)
              map audio_file keys matching *key* to feature keys
              for each feature_file:
               insert timbre-channel keys, powers, features into rTupleList
               Perform PLCA using self.num_components
               Perform clustering on collection of GT W functions (classify only if train=False)
               Sum H functions into channels by per-segment cluster assignments
               Write features as c timbre-channel feature files replacing original feature file                     
               Write Z functions as '.probs'
        """
        ext = self.uid + wave_suffix
        if self._test_timbre_channel_cache(key, wave_suffix):
            return True
        self.insert_audio_files(glob.glob(self.collection_path + os.sep + "*" + keyrepl + ext))
        aList, fList, pList, kList = self._get_extract_lists(key+ext, keyrepl+ext, wave_suffix)
        if len(aList) != self.num_patterns:
            print "audio_list length != num_patterns"
            raise TestCollectionError()
        W, Z, H = self._separate_components(fList)
        # clustering and assignment
        print "Kmeans clustering..."
        X = pylab.hstack(W)
        if train:
            assigns = self._cluster_components(X)
        else:
            assigns = self._assign_components_to_clusters(X)
        self._reconstruct_timbre_channels_features(fList, key, assigns, W, Z, H, self.feature_params['time_funs_only'])
        self.audio_collection.clear() # clear the audio_collection queue
        return True

    def _test_timbre_channel_cache(self, key, wave_suffix=".wav"):
        """
        ::

            Sweep over the expected timbre channel file names.
            Returns:
              True if all exist
              False if none exist
              Error if some exist
        """
        num_exist = 0
        for i in range(1, self.num_patterns): # skip 0 due to source file name clash
            for t_chan in range(1, self.feature_params['num_timbre_channels']): # skip 0 due to source file name clash
                f = None
                wav_name = self.collection_path + os.sep + "%03d"%i + "%03d"%t_chan + key + self.uid + wave_suffix
                try:
                    f = open(wav_name, 'r')
                    num_exist += 1
                except:
                    break
                finally:
                    if f:
                        f.close()
        if not num_exist:
            return False
        if num_exist == (self.num_patterns - 1) * (self.feature_params['num_timbre_channels'] - 1): # only checking timbre-channels >0
            return True
        else:
            print "Only partial cache exists of files with key", key
            raise TestCollectionError()
        
    def _separate_components(self, feature_list):
        """
        ::

            Perform PLCA extraction on incoming list of feature_files
            Returns lists of components, one per feature file:
            W, Z, H - plca component lists
            New :
              pre-conditions:
               feature_params[...] -> basename#hash#count#uid.features_ext
               *#hash#count#uid.features_ext -> feature_list
               _this_function
               for feature_file in feature_list:
                   V = load_feature(feature_file)
                   w,z,h = plca(V, syn_params['num_components']
                   W.append(w), Z.append(Z), H.append(H)
               return (W,Z,H)
               --
               assigns = cluster(W)           
               reconstruct per timbre-channel feature
               write as basename#channel#hash#count#uid.features_ext
        """
        # PLCA components extraction
        W, Z, H = [], [], []
        print "PLCA analysis..."
        for feature_file in feature_list:
            V = self._load_features(feature_file)
            w, z, h, norm, recon, logprob = plca.PLCA.analyze(V.T, self.syn_params['num_components'])
            W.append(w)
            Z.append(z)
            H.append(h)
        return (W, Z, H)

    def _load_features(self, feature_file):
        """
        ::

            Load an AudioDB feature file by filename
            Override _load_features to use features in different formats.
        """
        X = audiodb.adb.read(feature_file)
        return X

    def _reconstruct_timbre_channels_audio(self, audio_list, key, assigns, W, Z, H, S, F, wave_suffix=".wav"):
        """ 
        ::

            invert timbre-channels to wav files
        """
        print "Timbre-channel grouping and synthesis..."
        dummy_signal= self._invert_plca(W[0][:,0], Z[0][0], H[0][0,:], S[0], F)
        signal_length = dummy_signal.size
        for i, audio_file in enumerate( audio_list ):
            for t_chan in range( self.feature_params['num_timbre_channels'] ):
                wav_name = self.collection_path + os.sep + "%03d"%i + "%03d"%t_chan + key + self.uid + wave_suffix
                sig = pylab.zeros(signal_length)
                for j in range( self.syn_params['num_components'] ):
                    if assigns[i][j] == t_chan and W[i].shape[1]: # test for possibly empty W[i]
                        sig += self._invert_plca(W[i][:,j], Z[i][j], H[i][j,:], S[i], F)
                sound.wav_write(sig, wav_name, self.syn_params['sr'])        
        return True

    def _reconstruct_timbre_channels_features(self, feature_list, key, assigns, W, Z, H, time_funs_only=False):
        """
        ::

            invert timbre channels to feature files (direct to features, no audio inversion)
            clear rTupleList, append fName, pName, kName
        """
        print "Timbre-channel grouping and feature synthesis..."
        if not time_funs_only:
            V0 = plca.PLCA.reconstruct(pylab.mat(W[0][:,0]).T,Z[0][0],pylab.mat(H[0][0,:]))
        else:
            V0 = H[0][0,:]
        feature_suffix = self._get_feature_suffix()
        self.rTupleList = []
        for i, audio_file in enumerate( feature_list ):
            for t_chan in range( self.feature_params['num_timbre_channels'] ):
                feat_name = self.collection_path + os.sep + "%03d"%i + "%03d"%t_chan + key + self.uid + feature_suffix
                power_name = feat_name.replace(feature_suffix, self.feature_params['power_ext'])
                key_name = feat_name.replace(feature_suffix, ".wav")
                self.rTupleList.append([feat_name, power_name, key_name])
                V_hat = pylab.atleast_2d(pylab.zeros((V0.shape)))
                Z_sum = 0.0
                for j in range( self.syn_params['num_components'] ):
                    if assigns[i][j] == t_chan and W[i].shape[1]: # test for possibly empty W[i]
                        if not time_funs_only:
                            V_hat += plca.PLCA.reconstruct(pylab.mat(W[i][:,j]).T, Z[i][j], pylab.mat(H[i][j,:]))
                        else:
                            V_hat += Z[i][j] * H[i][j,:] # probabilistic weighted sum of H's
                        Z_sum += Z[i][j]
                if not time_funs_only:
                    V_hat = V_hat.T # convert spectral view to observation matrix
                audiodb.adb.write(feat_name, V_hat) 
                audiodb.adb.write(power_name, pylab.ones((V_hat.shape[0],1)) * Z_sum) # align probs to features
        return True


    def _cluster_components(self, X):
        """
        ::

            Cluster feature nd-arrays list, W, into self.feature_params['num_timbre_channels'] clusters
            return True on success
        """
        self.km = classifier.KMeans(self.feature_params['num_timbre_channels'])
        assigns = self.km.train(X.T)
        return assigns.reshape(-1, self.syn_params['num_components'])

    def _assign_components_to_clusters(self, X):
        """
        ::

            return the assignments for each observation
        """
        assigns = self.km.classify(X.T)
        return assigns.reshape(-1, self.syn_params['num_components'])
        
    def _invert_plca(self, w, z, h, s, f):
        """
        ::

            PLCA single-component reconstruction
            x =  ifft ( cqt.T x (w x z x h.T) x exp(angle(fft(a))) )
        """
        f.STFT = s
        V_hat = plca.PLCA.reconstruct(pylab.mat(w).T,z,pylab.mat(h))
        f.icqft(V_hat)
        return f.x_hat
