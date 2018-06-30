from models import forward
from utils import binarize_adj, renormalize_adj, row_renormalize_adj

import scipy.io as sio 
import os
import numpy as np


output_n = 16
DIR = '/mnt/disk1/cifar10'
def load(dataset):
    dbname = {
            'cifar10': 'cifar10_gist.mat',
            'NUSWIDE': 'nus_wide.mat',
         } 
    db = sio.loadmat(os.path.join(DIR, dbname[dataset]))

    trans_aff_matrix = sio.loadmat(os.path.join(DIR,
                                  'trans_aff_matrix.mat'))['trans_aff_matrix']
    adj = sio.loadmat(os.path.join(DIR, 'adj_matrix_5.mat'))['A']

    trans1 = row_renormalize_adj(binarize_adj(trans_aff_matrix))
    adj = renormalize_adj(binarize_adj(adj))
    trans2 = row_renormalize_adj(binarize_adj(trans_aff_matrix.T))

    traindata = db['traindata']
    testdata = db['testdata']
    return traindata, testdata, trans1, trans2, adj

traindata, testdata, trans1, trans2, adj = load('cifar10')
fn1 = forward(traindata, output_n, '/mnt/disk1/cifarmodel_{}'.format(output_n),
              adj, adj, adj)
fn2 = forward(testdata, output_n, '/mnt/disk1/cifarmodel_{}'.format(output_n), adj, trans1, trans2)

B = np.sign(fn1.infer())
tB = np.sign(fn2.infer())
sio.savemat('cifarv3_%d' % output_n, {'B': B, 'tB': tB})    
