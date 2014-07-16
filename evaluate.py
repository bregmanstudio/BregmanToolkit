# Evaluate.py -- methods for evaluating retrieval against a ground truth
# Bregman - python toolkit for music information retrieval

__version__ = '1.0'
__author__ = 'Michael A. Casey'
__copyright__ = "Copyright (C) 2010  Michael Casey, Dartmouth College, All Rights Reserved"
__license__ = "gpl 2.0 or higher"
__email__ = 'mcasey@dartmouth.edu'

import distance
import error
import audiodb
import pylab
import pickle
import sys
import tempfile

class Evaluator:
    """
    ::

        A class for evaluation of an audioDB (adb) database instance
        Evaluator initialization:
         required parameters:
            adb - the full path of an audioDB database instance
            ground_truth - a list of database indices for a class of query
         optional parameters:
            seq_length OR query_duration - length of queries w.r.t. adb time stamps
            tempo OR tempo_range - tempo/tempi over which to search
    """
    
    def __init__(self, adb, ground_truth, seq_length=None, query_duration=None, tempo=None, tempo_range=None):
        """
        ::

            Evaluator initialization:
            required parameters:
               adb - the full path of an audioDB database instance
               ground_truth - a list of database indices for a class of query
             optional parameters:
               seq_length OR query_duration - length of queries w.r.t. adb time stamps
               tempo OR tempo_range - tempo/tempi over which to search
        """
        if not adb:
            print "You must supply a valid audioDB database instance"
        if not ground_truth:
            print "You must supply a ground truth as a list of database indices"
        self.adb=None
        self.ground_truth=None
        self.set_adb(adb)
        self.set_ground_truth(ground_truth)
        self.seq_length=None
        self.query_duration=None
        self.tempo=None
        self.tempo_range=None
        self.set_optional_parameters(seq_length, query_duration, tempo, tempo_range)
        self.ranks_by_rank = None
        self.ranks_by_dists = None

    def set_adb(self, adb):
        """ 
        ::

            set the audioDB instance for this evaluator
            adb - the full path of an audioDB database instance
        """
        if self.adb!=None:
            del self.adb
        self.adb = adb

    def set_ground_truth(self, ground_truth):
        """ 
        ::

            set the ground truth for this instance
            ground_truth - a list of database indices for a class of query
        """
        if self.ground_truth!=None:
            del self.ground_truth
        self.ground_truth=ground_truth

    def set_optional_parameters(self, seq_length, query_duration, tempo, tempo_range):
        """        
        ::

            set optional parameters for evaluation:
              seq_length  - length of query in feature frames 
              query_duration - length of query in seconds
              tempo - change tempo of query by this proportion: e.g. 1.2 is 20% faster
              tempo_range - a list of tempo change queries to sweep
        """ 
        if seq_length:
            self.seq_length=seq_length
        elif query_duration:
            self.query_duration=query_duration
        if tempo:
            self.tempo=tempo
        elif tempo_range !=None:
            self.tempo_range=tempo_range

    def run(self,seq_length=None, query_duration=None, tempo=None, tempo_range=None):
        """
        ::

            Run the evaluation using optional parameters
              query_duration - length of query in seconds
              tempo - change tempo of query by this proportion: e.g. 1.2 is 20% faster
              tempo_range - a list of tempo change queries to sweep
        """
        self.set_optional_parameters(seq_length, query_duration, tempo, tempo_range)
        if self.tempo_range!=None:
            r,d=self.evaluate_tempo_range(seq_length=self.seq_length, 
                                          query_duration=self.query_duration, tempo_range=self.tempo_range)
        else:
            r,d=self.evaluate(seq_length=self.seq_length, 
                              query_duration=self.query_duration, tempo=self.tempo)
        self.ranks_by_rank = r
        self.ranks_by_dists = d

    def report(self):
        """
        ::

            Print the mean rank between query runs (at each tempo) and by pure distance
        """
        if self.ranks_by_rank !=None:
            print "Mean Rank by Minimum Rank = ", self.ranks_by_rank.mean()
        if self.ranks_by_dists !=None:
            print "Mean Rank by Distance = ", self.ranks_by_dists.mean()

    def evaluate(self, seq_length=None, query_duration=None, tempo=1.0, gt_only=True):
        """ 
        ::

            Evaluate loop over ground truth: query_duration varies with respect to tempo:
              query_duration - fractional seconds (requires adb.delta_time)
               OR seq_length - integer length of query sequence
              gt_only = if True, return only ground-truth results otherwise return full database results
        """
        if not tempo: tempo=1.0
        seq_length=self.set_seq_length(seq_length, query_duration)
        lzt_keys, lzt_lengths = self.get_adb_lists()
        ranks = pylab.ones( (len(self.ground_truth),len(lzt_keys)) )*float('inf')
        dists = pylab.ones( (len(self.ground_truth),len(lzt_keys)) )*float('inf')
        gt_list, gt_orig = self.initialize_search(seq_length, tempo)
        gt_orig_keys, gt_orig_lengths = zip(*gt_orig)
        gt_keys, gt_lengths = zip(*gt_list)

        # Loop over ground truth keys
        self.adb.configQuery['seqLength']=seq_length
        for i,q in enumerate(gt_keys):
            # Search
            if tempo==1.0:
                res = self.adb.query(key=q).rawData
            else:
                res = audiodb.adb.tempo_search(db=self.adb, Key=q, tempo=tempo)
            r_keys, r_dists, q_pos, r_pos = zip(*res)
            q_idx = gt_orig_keys.index(q)
            for r_idx, s in enumerate(lzt_keys):
                try:
                    k = r_keys.index(s)
                    ranks[q_idx][r_idx] = k
                    dists[q_idx][r_idx] = r_dists[k]
                except ValueError:
                    # print "Warning: adb key ", s, "not found in result."
                    pass
        self.ranks = ranks
        self.dists = dists
        if gt_only:
            ranks, dists = self.reduce_evaluation_to_gt(ranks, dists, query_duration=query_duration)    
        return ranks, dists

    def evaluate_tempo_range(self, seq_length=None, query_duration=None, tempo_range=[1.0], gt_only=True):
        """
        ::

            Loop over tempo range evaluating and integrating results:
             query_duration = fractional seconds (requires adb.delta_time)
              OR seq_length - integer length of query sequence
             tempo_range = list of tempo proportions relative to nominal 1.0
             gt_only = if True, return only ground-truth results otherwise return full database results        
        """
        seq_length=self.set_seq_length(seq_length, query_duration)
        out_ranks = []
        out_dists = []
        for t in tempo_range:
            print "Evaluating tempo: ", t
            # perform full evaluation, gt_only=False, for across-tempo merged ranks of gt in db results
            ranks, dists = self.evaluate(query_duration=query_duration, tempo=t, gt_only=False)
            out_ranks.append(ranks)
            out_dists.append(dists)
        out_ranks = pylab.dstack(out_ranks).min(2) # collapse runs
        out_dists = pylab.dstack(out_dists).min(2)
        if gt_only:
            out_ranks, out_dists = self.reduce_evaluation_to_gt(out_ranks, out_dists, query_duration=query_duration)    
        return out_ranks, out_dists

    def reduce_evaluation_to_gt(self, out_ranks, out_dists, seq_length=None, query_duration=None):
        """
        ::

            Pick out only ground truth from rank results:
             query_duration - fractional seconds (requires adb.delta_time)
              OR seq_length - integer length of query sequence
        """
        seq_length=self.set_seq_length(seq_length, query_duration)
        gt_orig_keys, gt_orig_lens = self.get_gt_lists()
        gt_keys, gt_lens = self.lower_bound_list_by_length(zip(gt_orig_keys, gt_orig_lens), seq_length)
        gt_row_idx = [gt_orig_keys.index(s) for s in gt_keys]
        ranks_gt = self.find_gt_ranks(out_ranks[gt_row_idx,:], gt_keys)
        dists_gt = self.find_gt_ranks(out_dists[gt_row_idx,:], gt_keys)
        return ranks_gt, dists_gt

    def set_seq_length(self, seq_length, query_duration):
        """
        ::

             check if we are using duration (in seconds) and set seq_length
             according to adb.delta_time, otherwise just use seq_length
        """
        if query_duration:
            seq_length = int( pylab.around( query_duration / self.adb.delta_time ) )
        if not seq_length:
            print "ERROR: You must specify a sequence length or query_duration"
            raise
        return seq_length

    
    def find_gt_ranks(self, out_ranks, ground_truth_keys=None):
        """
        ::
        
            Return ranks matrix for ground-truth columns only
        """
        r = out_ranks.argsort()
        lzt_keys, lzt_len = self.get_adb_lists()
        gt_idx = [lzt_keys.index(s) for s in ground_truth_keys]
        ranks = pylab.zeros((len(gt_idx),len(gt_idx)))
        for i in pylab.arange(len(gt_idx)):
            for j in pylab.arange(len(gt_idx)):
                ranks[i][j]=pylab.nonzero(r[i]==gt_idx[j])[0][0]
        return ranks

    def initialize_search(self, seq_length, tempo=1.0):
        """
        ::

            Initializes the evaluation loop search parameters
             build sequence length lower-bound list of included GT items >= seq_length
             build sequence length upper-bound list of excluded database items < seq_length
             set adb.configQuery parameters based on seq_length, tempo, and ground_truth
            returns gt_lower_bound_list, gt_orig_list
        """
        if tempo!=1.0:
            seq_lower_bound=int(pylab.around(seq_length/tempo))
        else:
            seq_lower_bound=seq_length
        print "sequence-lower-bound = ", seq_lower_bound
        gt_orig_keys, gt_orig_len = self.get_gt_lists()
        gt_orig = zip(gt_orig_keys, gt_orig_len)
        gt_list_keys, gt_list_len = self.lower_bound_list_by_length(gt_orig, seq_lower_bound, tempo)
        gt_list = zip(gt_list_keys, gt_list_len)
        print "GT query / retrieval list length = ", len(gt_list_keys)
        excl_keys, excl_lengths = self.upper_bound_list_by_length(self.adb.liszt(), seq_lower_bound)
        print "Database exclude list length = ", len(excl_keys)
        if len(excl_keys):
            self.adb.configQuery['excludeKeys']=list(excl_keys)
        else:
            self.adb.configQuery['excludeKeys']=[]
        self.adb.configQuery['seqStart']=0
        self.adb.configQuery['seqLength']=seq_length
        self.adb.configQuery['accumulation']='track'
        self.adb.configQuery['distance']='euclidean'
        self.adb.configQuery['radius']=0.0
        self.adb.configQuery['ntracks']=len(self.adb.liszt())
        self.adb.configQuery['npoints']=1
        if not self.adb.configCheck():
            print "Invalid query configuartion"
            raise    
        return gt_list, gt_orig

    def get_adb_lists(self):
        """
        ::

            return two lists of database keys, lengths
        """
        lzt = self.adb.liszt()
        lzt_keys, lzt_lens=zip(*lzt)
        return lzt_keys, lzt_lens

    def get_gt_lists(self):
        """
        ::

            return two lists of ground truth keys, lengths
        """
        lzt = self.adb.liszt()
        gt_list = [lzt[i] for i in self.ground_truth] 
        gt_keys, gt_lens=zip(*gt_list)
        return gt_keys, gt_lens

    def lower_bound_list_by_length(self,lst, length, tempo=1.0):
        """
        ::

            truncate (key, length) tuples by lower bound len
            return two lists incl_keys, incl_lens
        """
        inc_list=[]
        if tempo!=1.0:
            length = length * tempo # the query
        for i,item in enumerate(lst):
                if item[1] >= length: inc_list.append(item)
        inc_keys, inc_lens = zip(*inc_list)
        return inc_keys, inc_lens

    def upper_bound_list_by_length(self, lst, length):
        """
        ::

            truncate (key, length) tuples by upper bound length
            return excl_keys, excl_lens
        """
        excl_list=[]
        excl_keys=[]
        excl_lens=[]
        for i,item in enumerate(lst):
            if item[1] < length : excl_list.append(item)
        if len(excl_list):
            excl_keys, excl_lens = zip(*excl_list)
        return excl_keys, excl_lens



class TimbreChannelEvaluator(Evaluator):
    """
    ::

        An evaluator class for timbre channels.

        Inputs:

        adb - an AudioDB database in timbre-channel layout:
           ttttt0c0: track#tc
               h0 p0
               h1 p0
               h2 p0
           ttttt0c1: track#tc
               h0 p0
               h1 p0
               h2 p0
           ...
        ground_truth - list of ground_truth key indices in range 0 ... ntracks-1
        timbre_channels - number of timbre channels in audioDB database
        delta_time - per-frame delta time [0.1s]
        seq_length - query sequence length (in frames)
        query_duration - alternative query sequence length (in seconds)
        tempo - relative tempo scale of search [2.0 = half speed, 1.0 = no scaling, 0.5 = double speed]
        tempo_range - range of relative tempos to search
    """
    def __init__(self, adb, ground_truth, timbre_channels, delta_time=None, seq_length=None, query_duration=None, tempo=None, tempo_range=None):
        Evaluator.__init__(self, adb, ground_truth, seq_length=seq_length, query_duration=query_duration, tempo=tempo, tempo_range=tempo_range)
        self.timbre_channels = timbre_channels
        self.delta_time = delta_time

    def initialize_search(self, t_chan, seq_length, tempo=1.0):
        """
        Initialize search parameters: include list for t_chan, exclude list for !t_chan,
        Key format is nnnnnncc where nnnnnn is the track id, and cc is the tc id.
        """
        adb = self.adb
        if tempo!=1.0:
            seq_lower_bound=int(pylab.around(seq_length/tempo))
        else:
            seq_lower_bound=seq_length
        #print "sequence-lower-bound = ", seq_lower_bound
        gt_orig_keys, gt_orig_len = self.get_gt_lists(t_chan)
        gt_orig = zip(gt_orig_keys, gt_orig_len)
        gt_list_keys, gt_list_len = self.lower_bound_list_by_length(gt_orig, seq_lower_bound, tempo)
        gt_list = zip(gt_list_keys, gt_list_len)
        #print "GT query / retrieval list length = ", len(gt_list_keys)
        tc_keys, tc_lens = self.get_adb_lists(t_chan)
        excl_keys, excl_lengths = self.upper_bound_list_by_length(zip(tc_keys, tc_lens), seq_lower_bound)
        #print "Database exclude list length = ", len(excl_keys)
        includeKeys=list(pylab.setdiff1d(tc_keys, excl_keys))
        adb.configQuery['absThres']=0.0 # We'll take care of probability threshold in distance
        adb.configQuery['accumulation']='track' # per-track accumulation
        adb.configQuery['npoints']=1 # closest matching shingle
        adb.configQuery['ntracks']=len(includeKeys) # How many tracks to report
        adb.configQuery['distance']='euclidean'
        adb.configQuery['radius']=0.0 
        adb.configQuery['seqLength']=seq_length
        adb.configQuery['seqStart']=0
        adb.configQuery['exhaustive']=True # all sub-sequences search
        adb.configQuery['hopSize']=1 # all sub-sequences search with hop 1
        adb.configQuery['includeKeys']=includeKeys # include the non GT_ITEMs in search
        adb.configQuery['excludeKeys']=[] #excludeKeys # exclude the GT_ITEM from search
        if not self.adb.configCheck():
            print "Invalid query configuartion"
            raise    
        return gt_list, gt_orig
        
    def evaluate(self, seq_length=None, query_duration=None, tempo=1.0, gt_only=True, ground_truth=None, res_name=None):
        """
        ::

            Evaluate ranks and distances for each ground-truth pattern for retrieval of other GT patterns.
            Overridden method from Evaluator for timbre channels
             Map distances and weights for each query in each channel
             Reduce using Bhattacharyya distance metric and base-line averaging metric.

             Sets self.ranks_list, self.dists_list
             Display ranked results as in print_results()

            Returns a tuple (ranks,dists)
            Each element of the tuple contains len(test_set_list) lists, 
             each list contains ground-truth ranks, and whole-dataset distances, of each test-set 
             in the test_set_list
        """
        if ground_truth is None:
            ground_truth=self.ground_truth
        if not tempo: tempo=1.0
        self.seq_length=self.set_seq_length(seq_length, query_duration)
        self.avg_ranks = []
        self.bhatt_ranks = []
        self.avg_dists = []
        self.bhatt_dists = []
        qkeys, qlens = zip(*self.adb.liszt())
        for gt_item in ground_truth:
            t_qkeys=[]
            t_ikeys=[]
            t_rkeys=[]
            t_dists=[]
            print "Evaluating gt_item: ", gt_item
            for t_chan in range( self.timbre_channels ):
                print "\tc ", t_chan
                qkey = qkeys[t_chan::self.timbre_channels][gt_item] # gt query timbre-channel key
                qlen = qlens[t_chan::self.timbre_channels][gt_item] # gt query timbre-channel len
                t_qkeys.append(qkey)
                self.initialize_search(t_chan, min([self.seq_length,qlen]), tempo) # setup per-timbre-channel include list
                t_ikeys.append(self.adb.configQuery['includeKeys']) # timbre-channel search keys
                res = self.adb.query(key=qkey).rawData # get the search results
                if len(res):
                    rkeys, dst, qpos, rpos = zip(*res)
                    t_rkeys.append(rkeys) # timbre-channel distance-sorted keys
                    t_dists.append(dst) # All result distances
                else:
                    print "Empty result list: ", qkey
                    raise error.BregmanError()
            avg_ranks, avg_dists = self.rank_by_distance_avg(t_qkeys, t_ikeys, t_rkeys, t_dists)
            bhatt_ranks, bhatt_dists = self.rank_by_distance_bhatt(t_qkeys, t_ikeys, t_rkeys, t_dists)
            self.avg_ranks.append(avg_ranks)
            self.avg_dists.append(avg_dists)
            self.bhatt_ranks.append(bhatt_ranks)
            self.bhatt_dists.append(bhatt_dists)
            if res_name is None:
                res_name = tempfile.mktemp(suffix=".dat",prefix="results_",dir=".")
            f = open(res_name, "w")
            pickle.dump((self.avg_ranks, self.avg_dists, self.bhatt_ranks, self.bhatt_dists), f)
            f.close()
        return True

    def rank_by_distance_avg(self, qkeys, ikeys, rkeys, dists):
        """
        ::

            Reduce timbre-channel distances to ranks list by ground-truth key indices
            Kullback distances
        """
        # timbre-channel search using pre-computed distances
        ranks_list = []
        t_keys, t_lens = self.get_adb_lists(0) 
        rdists=pylab.ones(len(t_keys))*float('inf')
        for t_chan in range(self.timbre_channels): # timbre channels
            t_keys, t_lens = self.get_adb_lists(t_chan) 
            for i, ikey in enumerate(ikeys[t_chan]): # include keys, results
                try: 
                    # find dist of key i for query
                    i_idx = rkeys[t_chan].index( ikey ) # lower_bounded include-key index
                    a_idx = t_keys.index( ikey ) # audiodb include-key index
                    # the reduced distance function in include_keys order
                    # distance is the sum for now
                    if t_chan:
                        rdists[a_idx] += dists[t_chan][i_idx]
                    else:
                        rdists[a_idx] = dists[t_chan][i_idx]
                except:
                    print "Key not found in result list: ", ikey, "for query:", qkeys[t_chan]
                    raise error.BregmanError()
        #search for the index of the relevant keys
        rdists = pylab.absolute(rdists)
        sort_idx = pylab.argsort(rdists)   # Sort fields into database order
        for r in self.ground_truth: # relevant keys
            ranks_list.append(pylab.where(sort_idx==r)[0][0]) # Rank of the relevant key
        return ranks_list, rdists

    def rank_by_distance_bhatt(self, qkeys, ikeys, rkeys, dists):
        """
        ::

            Reduce timbre-channel distances to ranks list by ground-truth key indices
            Bhattacharyya distance on timbre-channel probabilities and Kullback distances
        """
        # timbre-channel search using pre-computed distances
        ranks_list = []
        t_keys, t_lens = self.get_adb_lists(0) 
        rdists=pylab.ones(len(t_keys))*float('inf')
        qk = self._get_probs_tc(qkeys)
        for i in range(len(ikeys[0])): # number of include keys
            ikey=[]
            dk = pylab.zeros(self.timbre_channels)
            for t_chan in range(self.timbre_channels): # timbre channels
                ikey.append(ikeys[t_chan][i])
                try: 
                    # find dist of key i for query
                    i_idx = rkeys[t_chan].index( ikey[t_chan] ) # dataset include-key match
                    # the reduced distance function in include_keys order
                    # distance is Bhattacharyya distance on probs and dists
                    dk[t_chan] = dists[t_chan][i_idx]
                except:
                    print "Key not found in result list: ", ikey, "for query:", qkeys[t_chan]
                    raise error.BregmanError()
            rk = self._get_probs_tc(ikey)
            a_idx = t_keys.index( ikey[0] ) # audiodb include-key index
            rdists[a_idx] = distance.bhatt(pylab.sqrt(pylab.absolute(dk)), pylab.sqrt(pylab.absolute(qk*rk)))
        #search for the index of the relevant keys
        rdists = pylab.absolute(rdists)
        sort_idx = pylab.argsort(rdists)   # Sort fields into database order
        for r in self.ground_truth: # relevant keys
            ranks_list.append(pylab.where(sort_idx==r)[0][0]) # Rank of the relevant key
        return ranks_list, rdists

    def _get_probs_tc(self, keys):
        """
        ::

            Retrieve probability values for a set of timbre-channel keys
        """
        pk = pylab.zeros(self.timbre_channels)
        for i,key in enumerate(keys):
            pk[i] = self.adb.retrieve_datum(key, powers=True)[0]
        return pk

    def get_adb_lists(self, tc):
        """
        ::

            return two lists of database keys, lengths for timbre channel tc
        """
        lzt = self.adb.liszt()
        lzt_keys, lzt_lens=zip(*lzt)
        return lzt_keys[tc::self.timbre_channels], lzt_lens[tc::self.timbre_channels]

    def get_gt_lists(self, tc):
        """
        ::

            return two lists of ground truth keys, lengths for timbre channel tc
        """
        lzt_keys, lzt_lens = self.get_adb_lists(tc)
        gt_keys = [lzt_keys[i] for i in self.ground_truth] 
        gt_lens = [lzt_lens[i] for i in self.ground_truth] 
        return gt_keys, gt_lens

    @staticmethod
    def prec_rec(ranks):
        """
        ::

            Return precision and recall arrays for ranks array data    
        """
        P = (1.0 + pylab.arange(pylab.size(ranks))) / ( 1.0 + pylab.sort(ranks))
        R = (1.0 + pylab.arange(pylab.size(ranks))) / pylab.size(ranks)
        return P, R

    @staticmethod
    def f_measure(ranks):
        """
        ::

            Return the standard F-measure for the given ranks
        """
        P,R = TimbreChannelEvaluator.prec_rec(ranks)
        return (2 * P * R) / (P + R)

