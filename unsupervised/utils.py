import numpy as np
import scipy.io as sio 
import scipy.sparse as sp
import os


def load_cifar10(dtype=np.float32, binary=True):
    _dirs = '/mnt/disk1/cifar10/'

    cifar10 = sio.loadmat(os.path.join(_dirs, 'cifar10_gist.mat'))
    
    traindata = dtype(cifar10['traindata'])
    testdata = dtype(cifar10['testdata'])
    cateTrainTest = cifar10['cateTrainTest'] 
    
    tradj, gfunc = load_data(_dirs)

    return traindata, testdata, cateTrainTest, tradj, gfunc

def load_imagenet(dtype=np.float32, binary=True) :
    _dirs = '/mnt/disk1/ImageNet1M/ImageNettiny/'

    sun = sio.loadmat(os.path.join(_dirs, 'sample.mat'))

    sample = dtype(sun['sample'])
    cateTrainTest = sun['cateTrainTest']

    sZ = sio.loadmat(os.path.join(_dirs, 'sample_anchor_graph.mat'))['Z']
    Z = sio.loadmat(os.path.join(_dirs, 'anchor_graph.mat'))['Z']
    tZ = sio.loadmat(os.path.join(_dirs, 'test_anchor_graph.mat'))['tZ']

    rowsum = np.array(Z.sum(axis=0)).flatten()
    rowsum = np.diag(1/rowsum)

    p1 = sZ.dot(rowsum)
    gfunc1 = Z.dot(p1.T)
    gfunc2 = tZ.dot(p1.T)

    tradj = sio.loadmat(os.path.join(_dirs, 'adj_matrix_5.mat'))['A']

    if binary:
        tradj = binarize_adj(tradj)

    tradj = renormalize_adj(tradj)
    
    return sample, cateTrainTest, tradj, gfunc1, gfunc2

def load_mnist(dtype=np.float32, binary=True):
    _dirs = '/mnt/disk1/mnist/'

    mnist = sio.loadmat(os.path.join(_dirs, 'mnist.mat'))

    traindata = dtype(mnist['traindata'])
    testdata = dtype(mnist['testdata'])
    cateTrainTest = mnist['cateTrainTest']

    tradj, ttadj, gfunc = load_data(_dirs)

    return traindata, testdata, cateTrainTest, tradj, ttadj, gfunc


def load_SIFT1M(dtype=np.float32, binary=True):
    _dirs = '/mnt/disk1/SIFT1M'

    SIFT1M = sio.loadmat(os.path.join(_dirs, 'SIFT1M.mat'))

    traindata = dtype(SIFT1M['traindata'])
    adj = sio.loadmat(os.path.join(_dirs, 'anchor_graph.mat'))['A']

    if binary:
        adj = binarize_adj(adj)
    adj = renormalize_adj(adj)    

    return traindata, adj

def load_nuswide(dtype=np.float32, binary=True):
    _dirs = '/mnt/disk1/NUS_WIDE'

    nuswide = sio.loadmat(os.path.join(_dirs, 'nus_wide'))

    traindata = dtype(nuswide['straindata'])
    cateTrainTest = dtype(nuswide['cateTrainTest'])

    sZ = sio.loadmat(os.path.join(_dirs, 'sample_anchor_graph.mat'))['Z']
    Z = sio.loadmat(os.path.join(_dirs, 'anchor_graph.mat'))['Z']
    tZ = sio.loadmat(os.path.join(_dirs, 'test_anchor_graph.mat'))['tZ']

    rowsum = np.array(Z.sum(axis=0)).flatten()
    rowsum = np.diag(1/rowsum)

    p1 = sZ.dot(rowsum)
    gfunc1 = Z.dot(p1.T)
    gfunc2 = tZ.dot(p1.T)

    import h5py
    dbs = h5py.File(os.path.join(_dirs, 'adj_matrix_5.mat'))

    A = dbs['A'].values()
    adj = convert_matlab_sparse_2_python(A)
    if binary:
        adj = binarize_adj(adj)
    adj = renormalize_adj(adj)    

    return traindata, cateTrainTest, adj, gfunc1, gfunc2

def load_data(_dirs, binary=True): 
    tradj = sio.loadmat(os.path.join(_dirs, 'adj_matrix_5.mat'))['A']
#    ttadj = sio.loadmat(os.path.join(_dirs, 'test_anchor_graph.mat'))['tA']

    if binary:
        tradj = binarize_adj(tradj)
#        ttadj = binarize_adj(ttadj) 

    tradj = renormalize_adj(tradj)
#    ttadj = renormalize_adj(ttadj)
    
    Z = sio.loadmat(os.path.join(_dirs, 'anchor_graph.mat'))
    tZ = sio.loadmat(os.path.join(_dirs, 'test_anchor_graph.mat'))

    Z = Z['Z']
    tZ = tZ['tZ']
    rowsum = np.array(Z.sum(axis=0)).flatten()
    rowsum = np.diag(1/rowsum)

    p1 = Z.dot(rowsum)
    gfunc = tZ.dot(p1.T)
    
    return tradj, gfunc

def convert_matlab_sparse_2_python(A):
    dim = len(A[2]) - 1
    return sp.csc_matrix((A[0], A[1], A[2]), shape=(dim, dim))

def save_matlab_sparse_matrix(filename, A):
    dim = len(A[2]) - 1
    np.savez(filename, data=A[0], indices=A[1], indptr=A[2], shape=(dim, dim), _type=sp.csc_matrix)

def save_sparse_matrix(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape, _type=array.__class__)

def load_sparse_matrix(filename):
    matrix = np.load(filename)

    _type = matrix['_type']
    sparse_matrix = _type.item(0)

    return sparse_matrix((matrix['data'], matrix['indices'],
                                 matrix['indptr']), shape=matrix['shape'])

def binarize_adj(adj):
    adj[adj != 0] = 1
    return adj
        
def renormalize_adj(adj):
    rowsum = np.array(adj.sum(axis=1))
    inv = np.power(rowsum, -0.5).flatten()
    inv[np.isinf(inv)] = 0.
    zdiag = sp.diags(inv)
    adj = sp.csr_matrix(adj)

    return adj.dot(zdiag).transpose().dot(zdiag)

def row_renormalize_adj(adj):
    rowsum = np.array(adj.sum(axis=1))
    inv = np.power(rowsum, -1).flatten()
    inv[np.isinf(inv)] = 0.

    d = sp.diags(inv).dot(adj)
    
    return sp.csr_matrix(d)

def sign_dot(data, func):
    return np.sign(np.dot(data, func))

def mAP(cateTrainTest, IX, num_return_NN=None):
    '''
        to-do: numpy c extension version.     
    '''
    numTrain, numTest = IX.shape

    num_return_NN = num_return_NN if num_return_NN else numTrain

    apall = np.zeros((numTest, 1))

    for qid in range(numTest):
        query = IX[:, qid]
        x, p = 0, 0

        for rid in range(num_return_NN):
            if cateTrainTest[query[rid], qid]:
                x += 1.0
                p += x/(rid*1.0 + 1.0)

        if not p: apall[qid] = 0.0
        else: apall[qid] = p/(x*1.0)

    return np.mean(apall)    

def topK(cateTrainTest, HammingRank, k=500):
    '''
        Deprecated
    '''
    numTest = cateTrainTest.shape[1]

    precision = np.zeros((numTest, 1))
    recall = np.zeros((numTest, 1))

    topk = HammingRank[:k, :]

    for qid in range(numTest):
        retrieved = topk[:, qid]
        rel = cateTrainTest[retrieved, qid]
        retrieved_relevant_num = np.sum(rel)
        real_relevant_num = np.sum(cateTrainTest[:, qid])

        precision[qid] = retrieved_relevant_num/(k*1.0)
        recall[qid] = retrieved_relevant_num/(real_relevant_num*1.0)

    return precision.mean(), recall.mean()

