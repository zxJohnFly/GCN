import theano
import theano.tensor as T
import numpy as np

from scipy.linalg import svd


class quan(object):
    def __init__(self, x, output_n, PCA=False):
        if PCA:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=output_n)
            B = np.sign(pca.fit_transform(x))
            self.B = theano.shared(np.float32(B))
        else:
            self.B = theano.shared(np.float32(np.sign(x)))

    def __call__(self, inputs):
        return theano.tensor.square(inputs - self.B)

    def update(self, x):
        self.B.set_value(np.float32(np.sign(x)))

    def get_R(self):
        return None

    def get_B(self):
        return None


class ITQ(object):
    def __init__(self, x, output_n, PCA=True, direct_infer=True):
        if PCA:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=output_n)
            x = pca.fit_transform(x)

        # orthogonal rotation
        R = np.random.randn(output_n, output_n)
        U, S, V = svd(R)
        self.R = theano.shared(np.float32(U))

        B = np.dot(x, U)
        self.B = theano.shared(np.float32(np.sign(B))) 
      
        self.update(x)
        self.db = np.sign(np.dot(self.get_B(), self.get_R().T))  
        self.direct_infer = direct_infer

    def __call__(self, inputs):
        if self.direct_infer:
            return self._direct(inputs)
        else:
            return self._undirect(inputs)

    def _undirect(self, inputs):
        a = T.dot(inputs. self.R)
        return theano.tensor.square(a - self.B)

    def _direct(self, inputs):
        return theano.tensor.square(self.db - inputs)

    def update(self, x, iter_n=50):
        B = self.B.get_value()
        R = self.R.get_value()

        for _ in range(iter_n):
            B = np.sign(np.dot(x, R))
            U, S, V = svd(np.dot(B.T, x))
            R = np.dot(V, U.T)

        self.B.set_value(np.float32(B))    
        self.R.set_value(np.float32(R))

    def get_R(self):
        return self.R.get_value()
 
    def get_B(self):
        return self.B.get_value()

    def set_R(self, B):
        self.B.set_value(B)

    def set_B(self, R):
        self.R.set_value(R)

class spectral_loss(object):
    def __init__(self, anchor_graph):
        #self.anchor_graph = anchor_graph
        zsum = anchor_graph.sum(axis=0)
        zdiag = np.diag(zsum.A1)
        zdiagn2 = zdiag ** -0.5
        zdiagn2[zdiagn2 == np.inf] = 0

        self.tr = anchor_graph.dot(zdiagn2)

    def __call__(self, inputs):
        return T.sum(T.square(T.dot(inputs.T, self.tr)))

