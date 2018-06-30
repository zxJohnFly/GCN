from __future__ import print_function

import lasagne
import theano
import theano.tensor as T

from layers import GraphConvLayer, FullyConnectedLayer
from lasagne.nonlinearities import rectify, tanh, identity
from objectives import ITQ, spectral_loss, quan

import numpy as np
import cPickle as pickle
import datetime
import os


class forward(object):
    def __init__(self, x, output_n, path, adj, trans_adj1, trans_adj2):
        self.input_var = T.fmatrix('x')
        self.x = x

        net = lasagne.layers.InputLayer(x.shape, self.input_var)
        net = GraphConvLayer(net, 512, nonlinearity=tanh, adjacency=trans_adj1)
        net = GraphConvLayer(net, 512, nonlinearity=tanh, adjacency=adj)
        net = GraphConvLayer(net, output_n, nonlinearity=tanh, adjacency=trans_adj2)

        self.net = net
        output = lasagne.layers.get_output(self.net)
        self._infer_fn = theano.function([self.input_var], output)

        self.load(path)

    def load(self, path):
        if not path:
            raise Exception('model file %s is not found.' % path)
        else:
            with open(path, 'r') as f:
                model = pickle.load(f)
            lasagne.layers.set_all_param_values(self.net, model)    
     

    def infer(self):
        return self._infer_fn(self.x) 
     

class GCN(object):
    def __init__(self, x, output_n, adj=None, anchor_graph=None):
        self.input_var = T.fmatrix('x')
        self.x = x
        
        net = lasagne.layers.InputLayer(x.shape, self.input_var)
        net = GraphConvLayer(net, 512, nonlinearity=tanh, adjacency=adj)
        net = GraphConvLayer(net, 512, nonlinearity=tanh, adjacency=adj)
        net = GraphConvLayer(net, output_n, nonlinearity=tanh, adjacency=adj)
       
        self.net = net  
        self.output_n = output_n
        self.anchor_graph = anchor_graph

    def build(self, lr=0.01, loss_fn='ITQ', PCA=True, optimizer=lasagne.updates.momentum, regularization=True):
        self.build_infer_fn()
        output = lasagne.layers.get_output(self.net)
        print('building loss function.')
        if loss_fn is 'ITQ':
            if PCA:
                x = self.x
            else:
                x = self.infer_fn(self.x)
            loss_fn = ITQ(x, self.output_n, PCA=PCA)
        elif loss_fn is 'spectral':    
            loss_fn = spectral_loss(self.anchor_graph)
        elif loss_fn is 'quan':
            if PCA:
                loss_fn = quan(self.x, self.output_n, PCA=PCA)
            else:
                x = self.infer_fn(self.x)
                loss_fn = quan(x, self.output_n, PCA=PCA)
        self.loss_fn = loss_fn

        loss = loss_fn(output).mean()

        if regularization:
            loss += 1e-4*lasagne.regularization.regularize_network_params(
                                        self.net, lasagne.regularization.l2) 
            
        self.lr = theano.shared(np.float32(lr))
        params = lasagne.layers.get_all_params(self.net, trainable=True)
        updates = optimizer(loss, params, learning_rate=self.lr)

        print('building train function.')
        self.train_fn = theano.function([self.input_var], loss, updates=updates)

    def build_infer_fn(self):
        output = lasagne.layers.get_output(self.net)
        self.infer_fn = theano.function([self.input_var], output)

    def get_infer_fn(self):
        return self.infer_fn

    def get_train_fn(self):
        return self.train_fn

    def get_R(self):
        return self.loss_fn.get_R()

    def get_B(self):
        return self.loss_fn.get_B()

    def save(self, path=None):
        model = lasagne.layers.get_all_param_values(self.net)
        if not path:
            time_stamp = datetime.datetime.now()
            path = os.path.join('./', '{}.pkl'.format(time_stamp.isoformat()))

        #R = self.get_R()
        #B = self.get_B()
        with open(path, 'w') as f:
            pickle.dump(model, f)
           # pickle.dump([model, R, B], f)

    def load(self, path):
        if not path:
            raise Exception('model file %s is not found.' % path)
        else:
            with open(path, 'r') as f:
                model = pickle.load(f)
            lasagne.layers.set_all_param_values(self.net, model)    


