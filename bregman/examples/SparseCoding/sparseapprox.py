"""
 sparseapprox - sparse code dictionary learning in python 
 
 Requires:
      sklearn - machine learning in python [http://scikit-learn.org/stable/]

2015, Michael A. Casey, Dartmouth College, Bregman Media Labs

License: 
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
http://creativecommons.org/licenses/by-nc/4.0/
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from bregman.suite import *
import os, sys, glob

from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.decomposition import ProjectedGradientNMF

class SparseApproxSpectrum(object):
    def __init__(self, n_components=49, patch_size=(8,8), max_samples=1000000, **kwargs):
        self.omp = OrthogonalMatchingPursuit()
        self.n_components = n_components
        self.patch_size = patch_size
        self.max_samples = max_samples
        self.D = None
        self.data = None
        self.components = None
        self.standardize=False

    def _extract_data_patches(self, X):
        self.X = X
        data = extract_patches_2d(X, self.patch_size)
        data = data.reshape(data.shape[0], -1)
        if len(data)>self.max_samples:
            data = np.random.permutation(data)[:self.max_samples]
        print data.shape
        if self.standardize:
            self.mn = np.mean(data, axis=0) 
            self.std = np.std(data, axis=0)
            data -= self.mn
            data /= self.std
        self.data = data

    def extract_codes(self, X, standardize=False):
        self.standardize=standardize
        self._extract_data_patches(X)
        self.dico = MiniBatchDictionaryLearning(n_components=self.n_components, alpha=1, n_iter=500)
        print "Dictionary learning from data..."
        self.D = self.dico.fit(self.data)
        return self

    def plot_codes(self, cbar=False, **kwargs):
        #plt.figure(figsize=(4.2, 4))
        N = int(np.ceil(np.sqrt(self.n_components)))
        kwargs.setdefault('cmap', pl.cm.gray_r)
        kwargs.setdefault('origin','bottom')
        kwargs.setdefault('interpolation','nearest')
        for i, comp in enumerate(self.D.components_):
            plt.subplot(N, N, i + 1)
            comp  = comp * self.std + self.mn if self.standardize else comp
            plt.imshow(comp.reshape(self.patch_size), **kwargs)
            if cbar:
                plt.colorbar()
            plt.xticks(())
            plt.yticks(())
        plt.suptitle('Dictionary learned from spectrum patches\n', fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    def extract_audio_dir_codes(self, dir_expr='/home/mkc/exp/FMRI/stimuli/Wav6sRamp/*.wav',**kwargs):
        flist=glob.glob(dir_expr)
        self.X = np.vstack([feature_scale(LogFrequencySpectrum(f, nbpo=24, nhop=1024).X,normalize=1).T for f in flist]).T
        self.D = extract_codes(self.X, **kwargs)
        self.plot_codes(**kwargs)
        return self

    def _get_approximation_coefs(self,data, components):
        w = np.array([self.omp.fit(components.T, d.T).coef_ for d in data])
        return w

    def reconstruct_spectrum(self, w=None, randomize=False):
        data = self.data
        components = self.D.components_
        if w is None:
            self.w = self._get_approximation_coefs(data, components)
            w = self.w
        if self.standardize:
            for comp in components: comp  = comp * self.std + self.mn
        if randomize:
            components = np.random.permutation(components)
        recon = np.dot(w, components).reshape(-1,self.patch_size[0],self.patch_size[1])
        self.X_hat = reconstruct_from_patches_2d(recon, self.X.shape)
        return self

    def reconstruct_individual_spectra(self, w=None, randomize=False, plotting=False, **kwargs):
        self.reconstruct_spectrum(w,randomize)
        w, components = self.w, self.D.components_
        self.X_hat_l = []
        for i in range(len(self.w.T)):
            r=np.array((np.matrix(w)[:,i]*np.matrix(components)[i,:])).reshape(-1,self.patch_size[0],self.patch_size[1])
            self.X_hat_l.append(reconstruct_from_patches_2d(r, self.X.shape))
        if plotting:
            plt.figure()            
            for k in range(self.n_components):
                plt.subplot(self.n_components**0.5,self.n_components**0.5,k+1)
                feature_plot(self.X_hat_l[k],nofig=1,**kwargs)
        return self

class NMFSpectrum(SparseApproxSpectrum):
    def __init__(self, **kwargs):
        SparseApproxSpectrum.__init__(self,**kwargs)

    def extract_codes(self, X, **kwargs):
        self.standardize=False
        self._extract_data_patches(X)
        kwargs.setdefault('sparseness','components')
        kwargs.setdefault('init','nndsvd')
        kwargs.setdefault('beta',0.5)
        print "NMF..."
        self.model = ProjectedGradientNMF(n_components=self.n_components, **kwargs)
        self.model.fit(self.data)        
        self.D = self.model
        return self

    def reconstruct_spectrum(self, w=None, randomize=False):
        if w is None:
            self.w = self.model.transform(self.data)
            w = self.w
        return SparseApproxSpectrum.reconstruct_spectrum(self, w=w, randomize=randomize)
