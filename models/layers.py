"""
Class definitions for layers used for building hybrid LMs.
"""

import numpy as np
import tensorflow as tf
from collections import deque


class PoolingWindow(object):
	_cntr = 0
	""" 
	Constructor for attention-based pooling window.
	
	ngram       - ngram value for which this pooling window is being used
	state_size  - list of LSTM state sizes, one for each layer
	output_size - length of output feature vector
	vec_size    - final word vector size (after concatenation)
	k           - size of sliding window over word vectors
	M           - maximum word length (width of word matrix)

	Creates and initializes variables for parameters.
	"""
	def __init__(self, num_features, state_sizes, vec_size, k, M):
		self.state_sizes = state_sizes
		self.num_features = num_features
		self.vec_size = vec_size
		self.max_word_len = M
		
		PoolingWindow._cntr += 1

		# dict to store all weights
		self.weights = {}

		with tf.variable_scope('PoolingWindow_' + str(PoolingWindow._cntr), reuse=tf.AUTO_REUSE):
			# state -> attention
			for i in range(len(self.state_sizes)):
				self.weights['W_l' + str(i+1)] = tf.get_variable(
					'W_l' + str(i+1), 
					[self.state_sizes[i], self.max_word_len], 
					initializer=tf.initializers.random_normal(stddev=1e-3), 
					trainable=True
				)
			# word vec -> attention
			for i in range(k):
				self.weights['W_v' + str(i+1)] = tf.get_variable(
					'W_v' + str(i+1),
					[self.vec_size, self.max_word_len],
					initializer=tf.initializers.random_normal(stddev=1e-3), 
					trainable=True
				)
			# bias
			self.weights['b'] = tf.get_variable(
				'b', 
				[1, self.max_word_len],
				initializer=tf.initializers.zeros,
				trainable=True
			)
			# gate; controls contribution of global info.
			self.weights['Wg'] = tf.get_variable(
				'Wg',
				[self.max_word_len, 1],
				initializer=tf.initializers.random_normal(stddev=1e-3), 
				trainable=True
			)
			self.weights['bg'] = tf.get_variable(
				'bg',
				[self.num_features, 1],
				initializer=tf.initializers.zeros,
				trainable=True
			)

	"""
	Creates an attention vector from current LSTM states, history of previously
	generated word vectors and currently generated 'raw' word matrix. Uses the 
	attention vector to build the final word vector.
	
	x           - the 'raw' word matrix i.e. conv features (shape = [batch_size, output_size, max_word_len])
	lstm_states - list of LSTM cell states (i^th element's shape = [batch_size, state_size[i]])
	"""
	def __call__(self, x, lstm_states, prev_word_vecs):
		# global attention features
		global_feats = self.weights['b']
		for i in range(len(self.state_sizes)):
			global_feats = global_feats + tf.matmul(lstm_states[i].c, self.weights['W_l'+str(i+1)])
		for i in range(len(prev_word_vecs)):
			global_feats = global_feats + tf.matmul(prev_word_vecs[i], self.weights['W_v'+str(i+1)])
		# normalize global_feats so that they have same mag. as local_feats
		# global_feats = global_feats/tf.norm(global_feats + 1e-9, axis=1, keepdims=True)

		# local attention features
		local_feats = tf.one_hot(tf.argmax(x, axis=2), self.max_word_len)

		# gate values
		gate_values = tf.nn.sigmoid(tf.einsum('ijk,kl->ijl', x, self.weights['Wg']) + self.weights['bg'])

		# combined features
		feats = gate_values*tf.expand_dims(global_feats, axis=1) + (1-gate_values)*local_feats

		# final attention vector
		attention = tf.nn.softmax(feats, dim=2)

		# final word vector
		return tf.reduce_sum(attention * x, axis=2)

class TransformationUnit(object):
	def __init__(self, input_dims, keep_prob):
		self._drop1 = tf.layers.Dropout(rate=1-keep_prob, noise_shape=[1, input_dims])
		self._dense1 = tf.layers.Dense(input_dims, activation=tf.nn.relu)
		self._dense2 = tf.layers.Dense(input_dims, activation=tf.nn.sigmoid)

		self._drop2 = tf.layers.Dropout(rate=1-keep_prob, noise_shape=[1, input_dims])
		self._dense3 = tf.layers.Dense(input_dims, activation=tf.nn.relu)
		self._dense4 = tf.layers.Dense(input_dims, activation=tf.nn.sigmoid)

	def __call__(self, inputs):
		dropped_inp = self._drop1(inputs)
		x1 = self._dense1(dropped_inp)
		t1 = self._dense2(dropped_inp)
		int_inp = x1*t1 + inputs*(1-t1)

		dropped_int_inp = self._drop2(int_inp)
		x2 = self._dense3(dropped_int_inp)
		t2 = self._dense4(dropped_int_inp)
		return x2*t2 + int_inp*(1-t2)

class ConvPoolLSTMUnit(object):
	def __init__(self, conv_units, pooling_windows, transformation_unit, lstm_cells, word_embedding, k):
		self.conv_units = conv_units
		self.pooling_windows = pooling_windows
		self.transformation_unit = transformation_unit
		self.lstm_cells = lstm_cells
		self.k = k
		self.word_embedding = word_embedding
		self.prev_word_vecs = deque([], maxlen=k)

	def __call__(self, input_list, states):
		x, word_idx = input_list
		inputs = tf.expand_dims(x, axis=3)
		feat_vecs = []
		for conv, pool in zip(self.conv_units, self.pooling_windows):
			output = tf.transpose(tf.squeeze(conv(inputs), axis=1), [0, 2, 1])
			output = pool(output, states, list(self.prev_word_vecs))
			feat_vecs.append(output)

		output = tf.concat(feat_vecs, axis=1)
		output = self.transformation_unit(output)
		self.prev_word_vecs.append(output)

		with tf.control_dependencies([tf.scatter_update(self.word_embedding, tf.squeeze(word_idx, axis=1), output)]):
			fstates = []
			for i, cell in enumerate(self.lstm_cells):
				with tf.variable_scope("layer_"+str(i+1)):
					output, fstate = cell(output, states[i])
					fstates.append(fstate)

		return output, fstates

	@property
	def output_size(self):
		return self.lstm_cells[-1].output_size

	@property
	def state_size(self):
		return [c.state_size for c in self.lstm_cells]

	def zero_state(self, batch_size, dtype):
		res = []
		for size in self.state_size():
			res.append(tf.contrib.rnn.LSTMStateTuple(
				tf.zeros([batch_size, size]), 
				tf.zeros([batch_size, size])
			))
		return res

	
	
