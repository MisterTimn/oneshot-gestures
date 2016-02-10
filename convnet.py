import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pickle
plt.ion()

import lasagne as nn

class convnet(object):

	def __init__(self):
        # define model: neural network
		l_in = nn.layers.InputLayer((None, 12, 64, 64))

		l_conv1 = nn.layers.Conv2DLayer(l_in, num_filters=4, filter_size=(5, 5))
		l_pool1 = nn.layers.MaxPool2DLayer(l_conv1, ds=(2, 2))

		l_conv2 = nn.layers.Conv2DLayer(l_pool1, num_filters=8, filter_size=(5, 5))
		l_pool2 = nn.layers.MaxPool2DLayer(l_conv2, ds=(2, 2))

		l_conv3 = nn.layers.Conv2DLayer(l_pool2, num_filters=16, filter_size=(4,4))
		l_pool3 = nn.layers.MaxPool2DLayer(l_conv3, ds=(2,2))

		l4 = nn.layers.DenseLayer(nn.layers.dropout(l_pool3, p=0.5), num_units=100)

		self.l_out = nn.layers.DenseLayer(l4, num_units=19, nonlinearity=T.nnet.softmax)

		self.L1 = 0.
		self.L2 = 0.0001
		params = nn.layers.get_all_non_bias_params(self.l_out)
		self.L1_term = sum(T.sum(T.abs_(p)) for p in params)
		self.L2_term = sum(T.sum(p**2) for p in params)

		objective = nn.objectives.Objective(self.l_out, loss_function = self.loss)

		cost_train = objective.get_loss()

		p_y_given_x = self.l_out.get_output(deterministic=True)
		y = T.argmax(p_y_given_x, axis=1)

		params = nn.layers.get_all_params(self.l_out)
		updates = nn.updates.nesterov_momentum(cost_train, params, learning_rate=0.001, momentum=0.9)

		# compile theano functions
		self.train = theano.function([l_in.input_var, objective.target_var], cost_train, updates=updates)

		self.predict = theano.function([l_in.input_var], y)

		self.test_output = theano.function([l_in.input_var], p_y_given_x)

	def log_loss(self, y, t, eps=1e-15):
	    """
	    cross entropy loss, summed over classes, mean over batches
	    """
	    y = T.clip(y, eps, 1 - eps)
	    loss = -T.sum(t * T.log(y)) / y.shape[0].astype(theano.config.floatX)
	    return loss

	def loss(self, y, t):
		"""
		loss function definition
		"""
		return self.log_loss(y, t) + self.L1 * self.L1_term + self.L2 * self.L2_term	

	def save_param_values(self, path):
		param_values = nn.layers.get_all_param_values(self.l_out)
		with open(path, 'wb') as f:
			pickle.dump(param_values,f)

	def load_param_values(self, path):
		with open(path, 'rb') as f:
			param_values = pickle.load(f)
		nn.layers.set_all_param_values(self.l_out,param_values)

	def train(self, x_batch, t_batch):
		return self.train(x_batch, t_batch)

	def predict(self, x_validate):
		return self.predict(x_validate)

	# def trunc_to(min,max,m):
	# 	for i in range(int(m.shape[0])):
	# 		np.clip(m[i], min, max, m[i])




