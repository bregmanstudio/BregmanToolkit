import classifier
import features
import sound
import numpy as np
import pylab as pl
import scipy.signal

class TimeSpan(object):   
    def __init__(self, start_time=None, end_time=None, duration=None):
        if start_time==None:
            raise ValueError("start_time must be provided")
        self.start_time=float(start_time)
        if end_time==None and start_time==None:
            raise ValueError("One of end_time or duration must be supplied")                
        self.duration = float(end_time) - self.start_time if duration is None else float(duration)
        self.end_time = self.start_time + float(duration) if end_time is None else float(end_time)
        if abs(self.end_time - self.start_time - self.duration)>np.finfo(np.float32).eps:
            raise ValueError("Inconistent end_time and duration provided")
    def __repr__(self):
        return "start_time=%.3f, end_time=%.3f, duration=%.3f"%(self.start_time, self.end_time, self.duration)

class Segment(object):
    def __init__(self, start_time=None, end_time=None, duration=None, features=None, label=0):
        self.time_span = TimeSpan(start_time, end_time, duration)
        self.features = [] if features is None else features
        self.label = str(label)
    def __repr__(self):
        return "(label=%s, %s, %s)"%(self.label, self.time_span.__repr__(), self.features.__repr__())

class Segmentation(object):
    """
    A segmentation consists of conjoined non-overlapping segments. 
    Each segment has a start_time, end_time, and implicit duration.
    
    A segmentation must be initialized either with a media filename or array (signal).
    """
    def __init__(self, media):
        self.media = media
        self.time_spans = []

    def time_spans_to_frames(self, span_list):
        pass

    def frames_to_time_spans(self, frame_list):
        pass
    
    def __getitem__(self, index):
        return self.time_spans[index]

    def __setitem__(self, index, segment):
        if type(segment) is not Segment:
            raise ValueError("Segmentation requires a Segment")
        self.time_spans[index]=segment

    def __len__(self):
        return len(self.time_spans)

    def append(self, segment):
        if type(segment) is not Segment:
            raise ValueError("Segmentation requires a Segment")
        self.time_spans.append(segment)

    def __repr__(self):
        return self.time_spans.__repr__()
        
class Segmentor(object):
    def __init__(self):
        pass
    def extract(self, media, num_clusers, feature, **kwargs):
        pass

class GeneralAudioSegmentor(Segmentor):
    def __init__(self):
        pass
    def extract(self, media, num_clusters, feature=features.LogFrequencyCepstrum, filter=True, **kwargs):
        """
        Given a media file (or signal) and num_clusters, return a segmentation.
        """
        self.media = media
        self.feature_params = kwargs
        self.num_clusters = num_clusters
        self.segmentation = Segmentation(media)
        self.F = feature(media, **kwargs)
        self.feature = self.F
        self.frame_rate = self.F.sample_rate / float(self.F.nhop)
        if filter:
            b,a = scipy.signal.filter_design.butter(8, .1)
            self.F.X = scipy.signal.lfilter(b,a,self.F.X,axis=1)
        self.km = classifier.KMeans(self.num_clusters)
        self.assigns = self.km.train(self.F.X.T)
        self.num_clusters = self.km.num_classes
        self.diffs = np.where(np.r_[1,np.diff(self.assigns),1])[0]
        seg_labels=self.assigns[self.diffs[:-1]]
        for i in range(len(self.diffs)-1):
            self.segmentation.append(Segment(self.diffs[i]/float(self.frame_rate), self.diffs[i+1]/float(self.frame_rate),label=seg_labels[i]))
        return self.segmentation

    def segmentation_plot(self, alpha=0.1, linewidth=1):
        """
        Display features and segmentation using different visualizations
        """
        pl.figure()
        pl.subplot(311)
        self.F.feature_plot(nofig=1)
        pl.subplot(312)
        self.cluster_plot(nofig=1)
        pl.subplot(313)
        self.linkage_plot(nofig=1, alpha=alpha, linewidth=linewidth)
        
    def cluster_plot(self, nofig=False):
        """
        Display segmentation clusters as indicator matrix
        """
        if not nofig:
            pl.figure()
        z = np.zeros((self.num_clusters, len(self.assigns)))
        for k in range(self.num_clusters): z[k, np.where(self.assigns==k)[0]]=k+1
        features.feature_plot(z,nofig=1)
        pl.title('Segmentation')
        pl.xlabel('Frames')            

    def linkage_plot(self, alpha=0.1, linewidth=1, nofig=False):
        """
        Display segmentation regions and linkage using arcs
        """
        if not nofig:
            pl.figure()
        ax = pl.gca()
        cols=['r','g','b','c','m','y','k']
        for k in range(self.num_clusters):
            kpos = np.where(self.assigns[self.diffs[:-1]]==k)
            for j in self.diffs[kpos]:
                for l in self.diffs[kpos]:
                    arc = pl.Circle(((l+j)/2.0,0.),radius=abs(l-j)/2.0, color=cols[k%len(cols)],fill=0, linewidth=linewidth, alpha=alpha)
                    ax.add_patch(arc)
        pl.show()
        pl.axis([0,self.F.X.shape[1],0,self.F.X.shape[1]/2.])
        pl.colorbar()

    def play_segs(self, k):
        """
        Play audio segments corresponding to cluster k (zero-based index)
        """
        if k is None:
            raise TypeError("play_segs requires an integrer cluster index: k")
        clusters = self.assigns[self.diffs[:-1]]
        sr = self.feature.sample_rate
        for seg in np.where(clusters==k)[0]:
            print self.segmentation[seg]
            x = sound.wavread(self.media, first=int(self.segmentation[seg].time_span.start_time*sr), last=int(self.segmentation[seg].time_span.duration*sr))
            sound.play(x[0].T, sr)

class SegmentationFeatureExtractor(object):
    def __init__(self):
        """
        Class to perform extraction operation on Segmentation objects
        """
        pass
    def extract(self, segmentation, feature, **kwargs):
        """
        Given a segmentation and a feature class (plus **kwargs), extract feature for each segment in the segmentation.
        """
        if type(segmentation.media) is not str:
            raise TypeError("Only file segmentation extraction is currently supported.")
        x,SR,fmt = sound.wavread(segmentation.media, last=1)
        for seg in segmentation: 
            x, xsr, fmt = sound.wavread(segmentation.media, 
                                       first=int(seg.time_span.start_time*SR), last=int(seg.time_span.duration*SR))
            f = feature(x, **kwargs)
            seg.features.append(f)

class ChapterSegmentor(Segmentor):
    pass

class SceneBoundarySegmentor(Segmentor):
    pass

class ShotBoundarySegmentor(Segmentor):
    pass

class MusicStructureSegmentor(Segmentor):
    pass

class AudioOnsetSegmentor(Segmentor):
    pass



