
# classifier - classification algorithms for Bregman toolkit
__version__ = '1.0'
__author__ = 'Michael A. Casey'
__copyright__ = "Copyright (C) 2010  Michael Casey, Dartmouth College, All Rights Reserved"
__license__ = "GPL Version 2.0 or Higher"
__email__ = 'mcasey@dartmouth.edu'

import numpy as N
import scipy.linalg
from random import random

class ClassifierError(Exception):
    pass

class Classifier:
    """
    ::

        Base class for supervised and unsupervised classifiers
    """ 
    def __init__(self, num_classes, max_iter=200, error_thresh=0.001, dist_fun='Euc'):
        self.num_classes=num_classes
        self.max_iter = max_iter
        self.error_thresh = error_thresh
        self.dist_fun = dist_fun
        self.M = None
        self.dists = None
        self.verbosity = 0

    def _initialize(self):
        pass

    def train(self, X, labels=None, reset=True):
        pass

    def classify(self, Y, labels=None):
        pass

    
class KMeans(Classifier):
    """
    ::

        Unsupervised classification using k-means and random initialization

        km = KMeans(num_classes, max_iter, error_thresh, dist_fun)
            num_classes - number of clusters to estimate
            max_iter - maximum number of iterations for training
            error_thresh - threshold for sum-square-differences of old/new means
            dist_fun - future parameter, to allow alternate metrics
            returns a new KMeans instance

        training:    
          assigns = train(X)
           X - numpy ndarray observation matrix, n x d, n observations, d dimensions

        after training:
          assigns = classify(X)
           X - numpy ndarray observation matrix, n x d, n observations, d dimensions

        self.M - the trained means
    """

    def __init__(self, num_classes, max_iter=200, error_thresh=0.001, dist_fun='Euc'):
        Classifier.__init__(self, num_classes, max_iter=200, error_thresh=0.001, dist_fun='Euc')

    def _initialize(self, data):
        """
        ::

            Initialize clusters semi-deterministically from data
        """

        #Cov = N.diag( N.diag( N.cov(data, rowvar=0) ) )
        k = self.num_classes
        #self.M = N.zeros((1,data.shape[1]))
        #self.M = N.dot(N.random.randn(k, data.shape[1]) , scipy.linalg.sqrtm(Cov).real) + N.dot(N.ones((k,1)) , data.mean(0).reshape(1,-1))
        self.M = N.random.randn(k, data.shape[1])
            
    def train(self, X, labels=None, reset=True):
        """
        ::

            Train the classifier using the data passed in X. 
            X is a row-wise observation matrix with variates in the columns
            and observations in the rows.
            If reset=True (default) means will be re-initialized from data.
        """
        self.X = X
        rw,cl = X.shape
        if reset:
            self._initialize(X)
        for i in range(self.max_iter):
            assignments = self.classify(self.X)
            sse = self._update_means(assignments)
            if sse < self.error_thresh:
                break
        return assignments

    def _update_means(self, assignments):
        """
        ::

            Given the assignment vector, compute new means
        """
        sse = N.zeros(self.num_classes)
        old_means = self.M.copy()
        empty_classes = []
        for k in range(self.num_classes):
            idx = N.where(assignments==k)[0]
            if len(idx):
                self.M[k,:] = self.X[idx,:].mean(0)
            else:
                empty_classes.append(k)
        if len(empty_classes):            
            self.M = self.M[N.setdiff1d(range(self.num_classes),empty_classes),:]
            old_means = old_means[N.setdiff1d(range(self.num_classes),empty_classes),:]
            self.num_classes -= len(empty_classes)
        sse = ((old_means - self.M)**2).sum()
        return sse

    def classify(self, Y, labels=None):
        """
        ::

            Given a trained classifier, return the assignments to classes for matrix Y.
        """
        self.dists = self._mtx_distance(self.M, Y)
        assignments = self.dists.argmin(0)
        return assignments

    @staticmethod # FIX ME, use bregman.distance functions
    def _mtx_distance(X,Y):
        """
        ::

            matrix-matrix distances between matrix X and matrix Y.
            Computes distances between every row of X and every row of Y

            Inputs:
            X, Y - a row-wise observation matrices

            Output:
            d (rX, rY), ndarray of squared distances between every row in X to every row in Y
        """
        d = N.zeros((X.shape[0], Y.shape[0]))
        for k in range(X.shape[0]):
            d[k,:] = ((N.kron(X[k,:],N.ones((Y.shape[0],1))) - Y)**2).sum(1)
        return d

class SoftKMeans(KMeans):
    """
    ::

        Employ soft kmeans algorithm for unsupervised clustering

        David MacKay,"Information Theory, Inference and Learning Algorithms", Cambridge, 2003
        Chapter 22

        Parameters:
           beta - softness/stiffness [2.0]
    """
    def __init__(self, num_classes, max_iter=200, error_thresh=0.001, beta = 2.0, dist_fun='Euc'):
        KMeans.__init__(self, num_classes, max_iter, error_thresh, dist_fun)
        self.beta = beta

    def _update_means(self, assignments):
        """
        Override KMeans _update_means to perform soft kmeans assignments
        """
        resp = N.exp(-self.beta * self.dists);
        resp /= N.dot(ones((self.num_classes,1)), resp.sum(0).reshape(1,-1))
        old_means = self.M.copy()
        for k in range(self.num_classes):
            Xk = N.dot(resp[k,:].reshape(-1,1), N.ones((1,self.X.shape[1]))) * self.X;
            self.M[k,:] = Xk.sum(0) / resp[k,:].sum(0)
        sse = ((old_means - self.M)**2).sum()
        if self.verbosity:
            print("sse = ", sse)
        return sse

    
class GaussianMulti(Classifier):
    """
    ::

        Supervised classification using a multivariate Gaussian model per class.
        Also known as a quadratic classifier (Therien 1989).

        gm = GaussianMulti(num_classes, max_iter, error_thresh, dist_fun)
            num_classes - number of clusters to estimate (required)
            max_iter - maximum number of iterations for training [200]
            error_thresh - threshold for sum-square-differences of old/new means [.001]
            dist_fun - future parameter, to allow alternate metrics [bregman.distance.euc]
            returns a new GaussianMulti instance

        training:    
          assigns = train(X, labels)
           X - numpy ndarray observation matrix, n x d, n observations, d dimensions
           labels - per row labels for data in X, must be same length as rows of X

        after training:
          assigns = classify(X)
           X - numpy ndarray observation matrix, n x d, n observations, d dimensions
           returns labels for class assignments to rows in X

        self.M - the trained means
        self.C - the trained covariances
    """
    def __init__(self, num_classes, max_iter=200, error_thresh=0.001, dist_fun='Euc'):
        Classifier.__init__(self, num_classes, max_iter, error_thresh, dist_fun)

    def train(self, data, labels=None, reset=True ):
        """
        ::

            myGM.train(data, labels)
               Supervised classification for each unique label in labels using data.
               Employs a multivariate Gaussian model per class.
               self.M - per-class Gaussian means
               self.C - per-class Gaussian covariance matrices
        """
        if labels is None:
            print("Supervised classifier needs labels to train.")
            raise ClassifierError()
        num_observations = data.shape[0]
        num_labels = labels.shape[0]
        labs = N.lib.arraysetops.unique(labels) # in lexicographic order
        self.labels = labs
        if len(labs) != self.num_classes:
            print("number of labels doesn't match number of classes in classifier instance")
            raise ValueError()
        self.M = N.zeros((len(labs),data.shape[1]))
        self.C = N.zeros((len(labs),data.shape[1],data.shape[1]))
        for k, c in enumerate(labs):
            c_idx = N.where(labels==c)[0] # logical index for label c in data
            self.M[k,:] = data[c_idx,:].mean(0)
            self.C[k,:,:] = N.cov(data[c_idx,:],rowvar=0)
    
    def classify(self, data, labels=None):
        """
        ::

            labels = myGM.classify(data)
        """
        probs = N.zeros((data.shape[0], self.num_classes))
        for k in range(self.num_classes):
            probs[:,k] = GaussianPDF(data, self.M[k,:], self.C[k,:,:])
        probs[N.where(probs<0)]=0
        assignments = N.argmax(probs,axis=1)
        return self.labels[assignments], probs

    def evaluate(self, data, labels):
        """
        ::
 
           Estimate predicted labels from data, compare with True labels.
            Returns:
                a - accuracy as a proportion: 0.0 - 1.0
        """
        predicted_labels, p = self.classify(data) 
        return len(N.where(predicted_labels.reshape(-1,1) == labels.reshape(-1,1))[0]) / float(len(labels))

    def classify_range(self, data, upper_bounds):
        """
        ::

            Classify data in ranges with given upper_bounds.
            The algorithm is a majority vote algorithm among the classes.

            Returns:
                a - assignments per upper_bound region
                c - counts of assignments per class            
        """
        start = 0
        assignments = N.zeros(len(upper_bounds))
        predicted_counts = N.zeros((len(upper_bounds),self.num_classes))
        predicted_labels, p = self.classify(data) 
        for i, stop in enumerate(upper_bounds):
            for j, label in enumerate(self.labels):
                predicted_counts[i,j] = len(N.where(predicted_labels[start:stop]==label)[0])
            assignments[i] = self.labels[N.argmax(predicted_counts[i,:])]
            start = stop
        return assignments, predicted_counts
    
    def evaluate_range(self, data, true_labels, upper_bounds):
        """
        ::

            Perform assignment aggregation within data ranges by majority vote.
            The maximum count among the K classes wins per range.
            In case of a tie, randomly select among tied classes.

            Returns:
                a - accuracy as a proportion: 0.0 - 1.0
                e - vector of True/False per range
        """
        start = 0
        evaluation = N.zeros(len(upper_bounds), dtype='bool')
        predicted_counts = N.zeros(self.num_classes)
        true_counts = N.zeros(self.num_classes)
        predicted_labels, p = self.classify(data)
        for i, stop in enumerate(upper_bounds):
            for j, label in enumerate(self.labels):
                predicted_counts[j] = len(N.where(predicted_labels[start:stop]==label)[0])
                true_counts[j] = len(N.where(true_labels[start:stop]==label)[0])
            evaluation[i] = self.labels[N.argmax(predicted_counts)] == self.labels[N.argmax(true_counts)]
            start = stop
        return len(N.where(evaluation)[0])/float(len(evaluation)), evaluation

def GaussianPDF(data, m, C):
    """
    ::

        Gaussian PDF lookup for row-wise data
        data - n-dimensional observation matrix (or vector)
        m    - Gaussian mean vector
        C    - Gaussian covariance matrix
    """
    n = C.shape[0]
    d = N.linalg.linalg.det(C)
    const = 1. / ( (2*N.pi)**(n/2) * N.sqrt(d) ) # assume equal priors
    g = lambda x: const * N.exp ( -0.5 * N.dot(N.dot((x - m), N.linalg.linalg.inv(C)), (x - m).T ) )
    p = [ g(x) for x in data]
    return N.array(p)

