import numpy as np
import forward_models.isoplanatic as isoplanatic

def get_forward_model(name='noise'):
    if name=='isoplanatic':
        return Isoplanatic
    else:
        return GaussianNoise

class Isoplanatic():

    def __init__(self, dr0=10, sig=0, speckle_flg=False, padsize=10):
        self.dr0 = np.float(dr0)
        self.sig = np.float(sig)
        self.speckle_flg = (speckle_flg == 'True')
        self.padsize = np.int(padsize)

    def apply(self,x):
        y,phi = isoplanatic.apply_aberration(x,self.dr0,self.sig,self.speckle_flg,self.padsize)
        return y

    def tostring(self):
        return 'Isoplanatic_dr0-%d_sig-%d_speckle-%s' % (self.dr0,self.sig,str(self.speckle_flg))


class GaussianNoise():

    def __init__(self, sigma=25):
        self.sigma = np.int(sigma)

    def apply(self,x):
        noise = np.random.randn(x.shape)*self.sigma/255.0
        y = x + noise
        return y

    def tostring(self):
        return 'GaussianNoise_sigma-%d' % self.sigma


    