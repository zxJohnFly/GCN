from utils import *
import h5py
import os
import numpy as np
import cPickle as pkl

import scipy.io as sio
import scipy.sparse as sp


def infer(model):
    W1, b1, W2, b2, W3, b3 = model

    def forward(adjs, x):
        print x.shape
        print W1.shape
        print b1.shape
        print adjs[0].shape
        x = np.tanh(adjs[0].dot(np.dot(x, W1) + b1))
        x = np.tanh(adjs[1].dot(np.dot(x, W2) + b2))
        x = np.tanh(adjs[2].dot(np.dot(x, W3) + b3))
        return np.sign(x)

    return forward

DIR = '/mnt/disk1/NUS_WIDE'
dbs = h5py.File(os.path.join(DIR, 'anchor_graph.mat'), 'r')
ptr = dbs['trans_aff_matrix']

traindata = sio.loadmat(os.path.join(DIR, 'NUS_WIDE.mat'))['traindata']
model, _, _ = pkl.load(open('/mnt/disk1/nuswidemodel_32')) 
gcn = infer(model)

adj = h5py.File(os.path.join(DIR, 'adj_matrix_5.mat'))
adj = adj['A'].values()
adj = convert_matlab_sparse_2_python(adj)
adj = renormalize_adj(binarize_adj(adj))

B = [] 
for i in xrange(ptr.shape[0]):
    print i
    beg = 0
    trans_adj = sp.csr_matrix(dbs[ptr[i][0]].value)
    adj1 =  row_renormalize_adj(binarize_adj(trans_adj))
    adj2 =  row_renormalize_adj(binarize_adj(trans_adj.T))
    
    x = traindata[beg:beg+trans_adj.shape[0],:]
    beg = beg + trans_adj.shape[0] + 1
    adjs = [adj1, adj, adj2] 
    B.append(gcn(adjs, x))

np.save('nuswidev3_32.npy', np.catenate(B))    
