import numpy as np
import forward_models.isoplanatic as isoplanatic
from abc import ABC, abstractmethod


class ForwardModel(ABC):

    @abstractmethod
    def apply(self,x):
        pass

    @abstractmethod
    def tostring(self):
        pass


class Isoplanatic(ForwardModel):

    @staticmethod
    def validate_args(dr0=10, sig=0, speckle_flg=False, padsize=10):
        d = dict()
        d['dr0'] = np.float(dr0)
        d['sig'] = np.float(sig)
        d['speckle_flg'] = (speckle_flg == 'True' or speckle_flg == True or speckle_flg == 1)
        d['padsize'] = np.int(padsize)
        return d

    def __init__(self, dr0=10, sig=0, speckle_flg=False, padsize=10):
        self.dr0 = dr0
        self.sig = sig
        self.speckle_flg = speckle_flg
        self.padsize = padsize

    def apply(self,x):
        y,phi = isoplanatic.apply_aberration(x,self.dr0,self.sig,self.speckle_flg,self.padsize)
        return y

    def tostring(self):
        return 'Isoplanatic_dr0-%d_sig-%d_speckle-%s' % (self.dr0,self.sig,str(self.speckle_flg))


class GaussianNoise(ForwardModel):

    @staticmethod
    def validate_args(sigma=25):
        d = dict()
        d['sigma'] = np.int(sigma)
        return d

    def __init__(self, sigma=25):
        self.sigma = sigma

    def apply(self,x):
        noise = np.random.randn(*x.shape)*self.sigma/255.0
        y = x + noise
        return y

    def tostring(self):
        return 'GaussianNoise_sigma-%d' % self.sigma


    