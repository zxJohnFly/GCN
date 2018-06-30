import lasagne
import theano.tensor as T

from lasagne import init
from lasagne import nonlinearities
from theano.sparse.basic import structured_dot

class GraphConvLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units,
                 nonlinearity=nonlinearities.rectify,
                 W=init.GlorotUniform(),
                 b=init.Constant(0.),
                 adjacency=None, **kwargs):
        super(GraphConvLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.num_units = num_units
        self.nonlinearity = nonlinearity
        self.W = self.add_param(W, (num_inputs, num_units), name='W')
        
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units, ), name='b',
                                    regularizable=False)

        # adjacency matrix for undirected graph
        self.adjacency = adjacency

    def get_output_for(self, input, **kwargs):
        conv = T.dot(input, self.W)
        # adjacency is a sparse matrix and conv is a 
        # dense matrix. 
        activation = structured_dot(self.adjacency, conv)

        if self.b is not None:
            activation = activation + self.b
        return self.nonlinearity(activation)    

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

class FullyConnectedLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, 
                 nonlinearity=nonlinearities.identity,
                 W=init.GlorotUniform(), 
                 b=init.Constant(0.), **kwargs):
        super(FullyConnectedLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.num_units = num_units
        self.nonlinearity = nonlinearity
        self.W = self.add_param(W, (num_inputs, num_units), name='W')
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units, ), name='b',
                                    regularizable=False)

    def get_output_for(self, input, **kwargs):
        activation = T.dot(input, self.W)

        if self.b is not None:
            activation = activation + self.b
        return self.nonlinearity(activation)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

