# AudioDB - routines for audio database I/O, searching, and manipulation
# Bregman - python toolkit for music information retrieval

__version__ = '1.0'
__author__ = 'Michael A. Casey'
__copyright__ = "Copyright (C) 2010  Michael Casey, Dartmouth College, All Rights Reserved"
__license__ = "GPL Version 2.0 or Higher"
__email__ = 'mcasey@dartmouth.edu'


# AudioDB libraries
import glob
import error
import pylab
import features

try:
    import pyadb
except ImportError:
    pass
from scipy.signal import resample
import pdb

class adb:
    """
    ::
    
        A helper class for handling audiodb databases
    """
    @staticmethod
    def read(fname, dtype='<f8'):
        """
        ::

            read a binary little-endien row-major adb array from a file.
            Uses open, seek, fread
        """
        fd = None
        data = None
        try:
            fd = open(fname, 'rb')
            dim = pylab.np.fromfile(fd, dtype='<i4', count=1)
            data = pylab.np.fromfile(fd, dtype=dtype)
            data = data.reshape(-1,dim)
            return data
        except IOError:
            print "IOError: Cannot open %s for reading." %(fname)
            raise IOError
        finally:
            if fd:
                fd.close()

    @staticmethod
    def write(fname,data, dtype='<f8'):
        """
        ::

            write a binary little-endien row-major adb array to a file.
        """
        fd = None
        rows,cols=data.shape
        try:
            fd = open(fname, 'wb')
            pylab.array([cols],dtype='<i4').tofile(fd)
            data = pylab.np.array(data,dtype=dtype)
            data.tofile(fd)
        except IOError:
            print "IOError: Cannot open %s for writing." %(fname)
            raise IOError
        finally:
            if fd:
                fd.close()

    @staticmethod
    def read_list(fileList):
        """
        ::

            read feature data from a list of filenames into a list of arrays
        """
        all_data = []
        #data = adb.read(fileList[0])
        #shape = data.shape
        for i,v in enumerate(fileList):
            all_data.append(adb.read(v))
        return all_data

    @staticmethod
    def read_dir(dir_expr):    
        """
        ::

            read directory of binary features with wilcards, e.g. dir('Features/*/*.mfcc')
            returns a list of observation matrices, one for each file in the wildcard search
        """
        fileList = glob.glob(dir_expr)
        return adb.read_list(fileList)

    @staticmethod
    def stack_vectors(data, win=1, hop=1, zero_pad=True):
        """
        ::
           create an overlapping stacked vector sequence from a series of vectors
            data - row-wise multidimensional data to stack
            win  - number of consecutive vectors to stack [1]
            hop  - number of vectors to advance per stack [1]
            zero_pad - zero pad if incomplete stack at end 
        """
        data = pylab.atleast_2d(data)
        nrows, dim = data.shape
        hop = min(hop, nrows)
        nvecs = nrows/int(hop) if not zero_pad else int(pylab.ceil(nrows/float(hop)))
        features = pylab.zeros((nvecs, win*dim))
        i = 0
        while i < nrows-win+1:
            features[i/hop,:] = data[i:i+win,:].reshape(1,-1)
            i+=hop
        if i/hop < nvecs:
            x = data[i::,:].reshape(1,-1)
            features[i/hop,:] = pylab.c_[x,pylab.zeros((1, win*dim - x.shape[1]))]
        return features

    @staticmethod
    def cov_vector(data, mn=True, cv=True, num_blocks=1, deltas=True, num_frames=None, norm_type=0, post_norm=0, frame_off=0, normalize=0):
        """
        ::

            turn time-series into Gaussian parameter vector consisting of mean and vectorized covariance
            data - multidimensional feature array
            mn - use means, True/False [True]
            cv - use covariances, True/False [True]
                if mn==False and cv==False, use raw data
            num_blocks - divide data into blocks, must be an integral divisor of rows [1] 
            deltas - use derivatives, True/False [True]
            num_frames - number of frames to keep, 0=all [0]
            norm_type - vector norming, 0=none, 1=L1, 2=L2, 3=infinity [0]
            post_norm - apply norming to output vectors (in addition to data vectors) [0]
            frame_off - offset start of data by frame_off [0]
            
            Returns:
               a vector corresponding to requested features
        """
        if num_frames:
            data = data[frame_off:frame_off+num_frames,:]
        nz_idx = abs(data).sum(1) > pylab.np.finfo(pylab.np.float32).eps
        if normalize:
            data = data - data.mean()
            data = data / abs(data).max()
        if norm_type<0: # zscore data
            norm_type = -norm_type
            data[nz_idx] -= data[nz_idx].mean(0)
            data[nz_idx] /= data[nz_idx].std(0)
        if norm_type==1:
            data[nz_idx] /= pylab.np.atleast_2d(abs(data[nz_idx]).sum(1)).T
        if norm_type==2:
            data[nz_idx] /= pylab.np.atleast_2d(pylab.np.sqrt((data[nz_idx]**2).sum(1))).T
        if norm_type==3:
            data[nz_idx] /= pylab.np.atleast_2d(abs(data[nz_idx]).max(1)).T
        data = data.reshape(num_blocks, -1, data.shape[1])
        if deltas:
            delta_data = pylab.np.zeros((data.shape[0],data.shape[1],data.shape[2]*2))
            for i in range(data.shape[0]):
                delta_data[i] = pylab.np.c_[data[i], pylab.np.vstack([pylab.np.zeros((1,data[i].shape[1])),pylab.np.diff(data[i],axis=0)])]
            data = delta_data
        if mn:
            M = list()
            for i in range(data.shape[0]):
                M.append(pylab.np.mean(data[i],0).reshape(-1))
            M = pylab.np.hstack(M)            
            features = M
        if cv:
            C = list()
            for i in range(data.shape[0]):
                C.append(pylab.np.cov(data[i],rowvar=0).reshape(-1))
            C = pylab.np.hstack(C)
            features = C
        if mn and cv:
            features = pylab.np.r_[M, C]
        if not mn and not cv:
            features = data.reshape(-1)
        if post_norm:
            if abs(features).sum() != 0: 
                if norm_type==1:
                    features /= abs(features).sum()
                if norm_type==2:
                    features /= pylab.np.sqrt((features**2).sum())
                if norm_type==3:
                    features /= abs(features).max()
        return features            

    @staticmethod
    def cov_vector_file(fname, **kwargs):
        """
        ::

            read time-series features from a file and convert to vectorized means and covariance matrix
        """
        data = adb.read(fname)
        features = adb.cov_vector(data, **kwargs)
        return features

    @staticmethod
    def cov_list(fileList, **kwargs):
        """
        ::

            read time-series features from a file list, convert to covariance vectors and stack as a matrix    
        """
        # sniff the first file
        data = adb.read(fileList[0])
        x_test = adb.cov_vector(data, **kwargs)
        shape = data.shape
        X = pylab.np.zeros((len(fileList),x_test.shape[0]))
        for i,v in enumerate(fileList):
            data = adb.read(v)
            X[i,:] = adb.cov_vector(data, **kwargs)
        return X

    @staticmethod
    def cov_dir(dir_expr, **kwargs):    
        """
        ::

            load features from wildcard search into vectorized covariance stacked matrix, one file per row
        """
        fileList = glob.glob(dir_expr)
        fileList.sort()
        return adb.cov_list(fileList, **kwargs)

    @staticmethod
    def resample_vector(data, prop):
        """
        ::

            resample the columns of data by a factor of prop e.g. 0.75, 1.25,...)
        """
        new_features = resample(data, pylab.around(data.shape[0]*prop))
        return new_features

    @staticmethod
    def sparseness(data):
        """
        ::

            Sparseness measure of row vector data.
            Returns a value between 0.0 (smooth) and 1.0 (impulse)
        """
        X = data
        if pylab.np.abs(X).sum() < pylab.np.finfo(pylab.np.float32).eps:
            return pylab.np.array([0.])
        r = X.shape[0]        
        s = (pylab.np.sqrt(r) - pylab.np.abs(X).sum(0)/pylab.np.sqrt((X**2).sum(0))) / (pylab.np.sqrt(r)-1)
        return s

    @staticmethod
    def get(dbname, mode="r"):
        """
        ::

            Retrieve ADB database, or create if doesn't exist
        """
        db = pyadb.Pyadb(dbname, mode=mode)
        stat = db.status()
        if not stat['l2Normed']:
            pyadb._pyadb._pyadb_l2norm(db._db)
        if not stat['hasPower']:
            pyadb._pyadb._pyadb_power(db._db)
        # make a sane configQuery for this db
        db.configQuery={'accumulation': 'track','distance': 'eucNorm','exhaustive': False,'falsePositives': False, 'npoints': 10,'ntracks': 10,'seqStart': 100, 'seqLength': 10, 'radius':0.4, 'absThres':-4.5, 'resFmt': 'list'}
        db.delta_time = 0.1 # feature delta time in seconds
        return db

    @staticmethod
    def insert(db, X, P, Key, T=None):
        """
        ::

            Place features X and powers P into the adb database with unique identifier given by string "Key"
        """
        db.insert(featData=X, powerData=P, timesData=T, key=Key)

    @staticmethod
    def search(db, Key):
        """
        ::

            Static search method
            returns sorted list of results
        """
        if not db.configCheck():
            print "Failed configCheck in query spec."
            print db.configQuery
            return None
        res = db.query(Key)
        res_resorted = adb.sort_search_result(res.rawData)
        return res_resorted

    @staticmethod
    def tempo_search(db, Key, tempo):
        """
        ::

            Static tempo-invariant search
            Returns search results for query resampled over a range of tempos.
        """
        if not db.configCheck():
            print "Failed configCheck in query spec."
            print db.configQuery
            return None
        prop = 1./tempo # the proportion of original samples required for new tempo
        qconf = db.configQuery.copy()
        X = db.retrieve_datum(Key)
        P = db.retrieve_datum(Key, powers=True)
        X_m = pylab.mat(X.mean(0))
        X_resamp = pylab.array(adb.resample_vector(X - pylab.mat(pylab.ones(X.shape[0])).T * X_m, prop))
        X_resamp += pylab.mat(pylab.ones(X_resamp.shape[0])).T * X_m
        P_resamp = pylab.array(adb.resample_vector(P, prop))
        seqStart = int( pylab.around(qconf['seqStart'] * prop) )
        qconf['seqStart'] = seqStart
        seqLength = int( pylab.around(qconf['seqLength'] * prop) )
        qconf['seqLength'] = seqLength
        tmpconf = db.configQuery
        db.configQuery = qconf
        res = db.query_data(featData=X_resamp, powerData=P_resamp)
        res_resorted = adb.sort_search_result(res.rawData)
        db.configQuery = tmpconf
        return res_resorted

    @staticmethod
    def sort_search_result(res):
        """
        ::

            Sort search results by stripping out repeated results and placing in increasing order of distance.
        """
        if not res or res==None:
            return None
        a,b,c,d = zip(*res)
        u = adb.uniquify(a)
        i = 0
        j = 0
        k = 0
        new_res=[]
        while k < len(u)-1:
            test_str = u[k+1]
            try:
                j = a.index(test_str,i)
            except ValueError:
                break
            tmp=res[i:j]
            tmp.reverse()
            for z in tmp: new_res.append(z) 
            i = j
            k += 1
        if j<len(res)-1:
            tmp=res[j:len(res)]
            tmp.reverse()
            for z in tmp: new_res.append(z) 
        return new_res

    @staticmethod
    def uniquify(seq, idfun=None): 
        """
        ::

            Remove repeated results from result list
        """
        # order preserving
        if idfun is None:
            def idfun(x): return x
        seen = {}
        result = []
        for item in seq:
            marker = idfun(item)
            if marker in seen: continue
            seen[marker] = 1
            result.append(item)
        return result

    @staticmethod
    def insert_audio_files(fileList, dbName, chroma=True, mfcc=False, cqft=False, progress=None):
        """
        ::

            Simple insert features into an audioDB database named by dbBame.
            Features are either chroma [default], mfcc, or cqft. 
            Feature parameters are default.
        """
        db = adb.get(dbName, "w")
        if not db:
            print "Could not open database: %s" %dbName
            return False    
        del db # commit the changes by closing the header
        db = adb.get(dbName) # re-open for writing data
        # FIXME: need to test if KEY (%i) already exists in db
        # Support for removing keys via include/exclude keys
        for a, i in enumerate(fileList):
            if progress:
                progress((a+0.5)/float(len(fileList)),i) # frac, fname
            print "Processing file: %s" %i
            F = features.Features(i)            
            if chroma: F.feature_params['feature']='chroma'
            elif mfcc: F.feature_params['feature']='mfcc'
            elif cqft: F.feature_params['feature']='cqft'
            else:
                raise error.BregmanError("One of chroma, mfcc, or cqft must be specified for features")
            F.extract()
            # raw features and power in Bels
            if progress:
                progress((a+1.0)/float(len(fileList)),i) # frac, fname
            db.insert(featData=F.CHROMA.T, powerData=adb.feature_scale(F.POWER, bels=True), key=i) 
            # db.insert(featData=F.CHROMA.T, powerData=F.feature_scale(F.POWER, bels=True), key=i)
        return db


    @staticmethod
    def insert_feature_files(featureList, powerList, keyList, dbName, delta_time=None, undo_log10=False):
        """
        ::

            Walk the list of features, powers, keys, and, optionally, times, and insert into database
        """
        
        if delta_time == None:
            delta_time = 0.0
        db = adb.get(dbName,"w")

        if not db:
            print "Could not open database: %s" %dbName
            return False    
        # FIXME: need to test if KEY (%i) already exists in db
        # Support for removing keys via include/exclude keys
        for feat,pwr,key in zip(featureList, powerList, keyList):
            print "Processing features: %s" %key
            F = adb.read(feat)
            P = adb.read(pwr)
            a,b = F.shape

            if(len(P.shape)==2):
                P = P.reshape(P.shape[0])

            if(len(P.shape)==1):
                c = P.shape[0]
            else:
                print "Error: powers have incorrect shape={0}".format(P.shape)
                return None

            if a != c:
                F=F.T
                a,b = F.shape
                if a != c:
                    print "Error: powers and features different lengths powers={0}*, features={1},{2}*".format(c, a, b)
                    return None
            # raw features, power in Bels, and key
            if undo_log10:
                F = 10**F
            if delta_time!=0.0:
                T = pylab.c_[pylab.arange(0,a)*delta_time, (pylab.arange(0,a)+1)*delta_time].reshape(1,2*a).squeeze() 
            else:
                T = None
            db.insert(featData=F, powerData=P, timesData=T, key=key)
        return db

