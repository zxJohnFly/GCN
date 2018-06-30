import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_double

matrix = npct.ndpointer(dtype=np.int32, ndim=2, flags='CONTIGUOUS')
vector = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')

libm = npct.load_library("libmetrics", ".")

libm.mAP.restype = c_double
libm.mAP.argtypes = [matrix, matrix, c_int, c_int, c_int, vector]

libm.topK.restype = None
libm.topK.argtypes = [matrix, matrix, c_int, c_int, c_int, vector, vector]

def mAP(cateTrainTest, IX, topk=None):
    cateTrainTest = np.ascontiguousarray(cateTrainTest, np.int32)
    IX = np.ascontiguousarray(IX, np.int32)

    m, n = cateTrainTest.shape
    m, n = np.int32(m), np.int32(n)

    if topk is None:
        topk = m
    mAPs = np.zeros(n, dtype=np.float64)    
    return libm.mAP(cateTrainTest, IX, topk, m, n, mAPs)   

def topK(cateTrainTest, IX, topk=500):
    cateTrainTest = np.ascontiguousarray(cateTrainTest, np.int32)
    IX = np.ascontiguousarray(IX, np.int32)

    m, n = cateTrainTest.shape
    m, n = np.int32(m), np.int32(n)

    precs = np.zeros(n, dtype=np.float64)
    recs = np.zeros(n, dtype=np.float64)

    libm.topK(cateTrainTest, IX, topk, m, n, precs, recs)

    return np.mean(precs), np.mean(recs)

