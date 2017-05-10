from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pickle
plt.ion()

import lasagne as nn

class convnet_oneshot(object):

    def __init__(self, num_output_units=20, num_layers_retrain=1):
        print("Initializing convnet model")
        # define model: neural network
        input_var = T.tensor4("inputs")
        target_var = T.ivector("targets")

        input = nn.layers.InputLayer(		shape=(None, 12, 64, 64),
                                            input_var=input_var	)

        conv1 = nn.layers.Conv2DLayer(	    input,
                                            num_filters=8, filter_size=(5, 5),
                                            nonlinearity=nn.nonlinearities.rectify,
                                            W=nn.init.GlorotUniform()	)
        pool1 = nn.layers.MaxPool2DLayer(	conv1, pool_size=(2, 2))

        conv2 = nn.layers.Conv2DLayer(	    pool1 ,
                                            num_filters=16, filter_size=(5, 5),
                                            nonlinearity=nn.nonlinearities.rectify	)
        pool2 = nn.layers.MaxPool2DLayer(	conv2, pool_size=(2, 2)	)

        conv3 = nn.layers.Conv2DLayer(	    pool2,
                                            num_filters=32, filter_size=(4,4),
                                            nonlinearity=nn.nonlinearities.rectify	)
        pool3 = nn.layers.MaxPool2DLayer(   conv3, pool_size=(2,2))

        self.dense1 = nn.layers.DenseLayer(	nn.layers.dropout(pool3, p=0.5),
                                            num_units=800,
                                            nonlinearity=nn.nonlinearities.rectify	)

        self.dense2 = nn.layers.DenseLayer(	nn.layers.dropout(self.dense1, p=0.5),
                                            num_units=100,
                                            nonlinearity=nn.nonlinearities.rectify	)


        self.dense_excluding = nn.layers.DenseLayer(self.dense2,
                                            num_units=num_output_units-1,
                                            nonlinearity=nn.nonlinearities.identity	)
        if(num_layers_retrain==2):
            self.dense_excluding = nn.layers.DenseLayer(self.dense2,
                                                        num_units=num_output_units - 1,
                                                        W=nn.init.Constant(0.0),
                                                        b=nn.init.Constant(0.0),
                                                        nonlinearity=nn.nonlinearities.identity)

        #w b initialiseren op 0
        self.dense_oneshot = nn.layers.DenseLayer(self.dense2,
                                                  num_units=1,
                                                  W=nn.init.Constant(0.0),
                                                  b=nn.init.Constant(0.0),
                                                  nonlinearity=nn.nonlinearities.identity )

        self.concat = nn.layers.ConcatLayer((self.dense_excluding,self.dense_oneshot))

#nonlinearity layer

        self.network = nn.layers.NonlinearityLayer( self.concat,
                                                    nonlinearity=nn.nonlinearities.softmax )


        L1 = 0.
        L2 = 0.0001
        l2_penalty = nn.regularization.regularize_network_params(self.network, nn.regularization.l2)
        # l1_penalty = nn.regularization.regularize_network_params(self.network, nn.regularization.l1)
        # params = nn.layers.get_all_params(self.network)
        # self.L1_term = sum(T.sum(T.abs_(p)) for p in params)
        # self.L2_term = sum(T.sum(p**2) for p in params)

        prediction  = nn.layers.get_output(self.network)
        loss = nn.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        loss = loss + l2_penalty * L2 #+ l1_penalty * L1


        # params = nn.layers.get_all_params(self.network, trainable=True)
        # params = [  param for param in self.dense2.get_params(trainable=True),
        #           param for param in self.network.get_params(trainable=True)]

        params = self.network.get_params(trainable=True)
        params.extend(self.concat.get_params(trainable=True))
        params.extend(self.dense_oneshot.get_params(trainable=True))
        if(num_layers_retrain>=2):
            params.extend(self.dense_excluding.get_params(trainable=True))
        if(num_layers_retrain>=3):
            params.extend(self.dense2.get_params(trainable=True))

        updates = nn.updates.nesterov_momentum(	loss, params,
                                                learning_rate=0.0001,
                                                momentum=0.9)

        test_prediction = nn.layers.get_output(	self.network, deterministic=True	)
        test_loss = nn.objectives.categorical_crossentropy(test_prediction, target_var)
        test_loss = test_loss.mean()

        test_acc = T.mean(	T.eq(	T.argmax(test_prediction, axis=1), target_var),
                            dtype=theano.config.floatX)

        output = T.argmax(test_prediction,axis=1)

        print('Compiling train function')
        self.train = theano.function([input_var, target_var], loss, updates=updates)

        print("Compiling validation function")
        self.validate = theano.function([input_var, target_var], [test_loss, test_acc])

        print("Functions compiled, convnet model initialized")
        self.test_output = theano.function([input_var], output)

    # def log_loss(self, y, t, eps=1e-15):
    #     """
    #     cross entropy loss, summed over classes, mean over batches
    #     """
    #     y = T.clip(y, eps, 1 - eps)
    #     loss = -T.sum(t * T.log(y)) / y.shape[0].astype(theano.config.floatX)
    #     return loss

    # def loss(self, y, t):
    # 	"""
    # 	loss function definition
    # 	"""
    # 	return self.log_loss(y, t) + self.L1 * self.L1_term + self.L2 * self.L2_term

    def preload_excluding_model(self, path):
        with open(path, 'rb') as f:
            param_values = pickle.load(f)
        nn.layers.set_all_param_values(self.dense_excluding,param_values)

    def save_param_values(self, path):
        param_values = nn.layers.get_all_param_values(self.network)
        with open(path, 'wb') as f:
            pickle.dump(param_values,f)

    def load_param_values(self, path):
        with open(path, 'rb') as f:
            param_values = pickle.load(f)
        nn.layers.set_all_param_values(self.network,param_values)

    def train(self, x_batch, y_batch):
        return self.train(x_batch, y_batch)

    def validate(self, x_validate, y_validate):
        return self.validate(x_validate, y_validate)

    # def trunc_to(min,max,m):
    # 	for i in range(int(m.shape[0])):
    # 		np.clip(m[i], min, max, m[i])




