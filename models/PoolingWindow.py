"""
charCNN-LSTM model that uses a mixture of word- and char-level embeddings.
"""

import tensorflow as tf
import numpy as np
import sys, time

from models.layers import PoolingWindow, TransformationUnit, ConvPoolLSTMUnit

tf.reset_default_graph()
np.random.seed(1)
tf.set_random_seed(1)

class Model(object):
	def __init__(self, config):
		""" Method for initializing model and constructing graph """
		self.graph = tf.Graph()
		with self.graph.as_default():
			# Placeholders for input tensors/matrices
			self._char_idx = tf.placeholder(tf.int32, [None, config.timesteps, config.max_word_length], name="char_input_indices")
			self._lengths = tf.placeholder(tf.int32, [None, config.timesteps], name="word_lengths")
			self._word_idx = tf.placeholder(tf.int32, [2, None, config.timesteps], name="word_indices")
			self._states = tf.placeholder(tf.float32, [config.n_layers, 2, None, config.num_units], name="lstm_states")

			# Parameters
			self._batch_size = tf.placeholder_with_default(20, shape=[], name="batch_size")
			self._lr = tf.placeholder_with_default(1.0, shape=[], name="learning_rate")
			
			# TensorFlow variables
			self.global_step = tf.Variable(0, trainable=False, name="global_step")
			self.best_metric = tf.Variable(100000.0, trainable=False, name="best_metric")
			self.new_best_metric = tf.placeholder(tf.float32, shape=[], name="new_best_metric")
			self.update_best_metric = tf.assign(self.best_metric, self.new_best_metric)
			self.epoch_cntr = tf.Variable(0, trainable=False, name="epoch_counter")
			self.incr_epoch = tf.assign_add(self.epoch_cntr, 1)

			# Other variables
			self.config = config
			inp_words, targets = tf.unstack(self._word_idx)
			inp_words = tf.expand_dims(inp_words, axis=2)

			# Create embeddings
			with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE, initializer=tf.initializers.random_uniform(-0.05, 0.05)):
				# Embedding containing char-level n-gram information
				self.input_word_embedding = tf.get_variable("M_c", [config.word_vocab_size, config.num_dims])
				# Word-level output embedding
				self.output_word_embedding = tf.get_variable("M_w", [config.word_vocab_size, config.word_dims])
				# Char-embedding for input convolutional layer
				self.char_embedding = tf.get_variable("char", [config.char_vocab_size, config.char_dims])

			###############################################################################################
			# Generating word vectors from constituent character vectors
			###############################################################################################

			# Extract character vectors from embedding
			with tf.variable_scope("extract_char_vectors", reuse=tf.AUTO_REUSE, initializer=tf.initializers.random_uniform(-0.05, 0.05)):
				char_vecs = tf.gather(self.char_embedding, self._char_idx, axis=0)
				char_vecs = tf.transpose(char_vecs, [0, 1, 3, 2])
				mask = tf.sequence_mask(self._lengths, maxlen=config.max_word_length, dtype=tf.float32)
				char_vecs = tf.tile(tf.expand_dims(mask, axis=2), [1, 1, config.char_dims, 1])*char_vecs

			# Apply convolution
			with tf.variable_scope("convolution", reuse=tf.AUTO_REUSE, initializer=tf.initializers.random_uniform(-0.05, 0.05)):
				self.conv_units = [tf.layers.Conv2D(
					filters=n_fts, 
					kernel_size=[config.char_dims, k],
					strides=[config.char_dims, 1],
					padding="same",
				) for k, n_fts in zip(config.kernel_sizes, config.kernel_features)]

				self.pooling_windows = [PoolingWindow(
					num_features=n_fts,
					state_sizes=[config.num_units for _ in range(config.n_layers)],
					vec_size=config.num_dims,
					k=config.sliding_window_size,
					M=config.max_word_length,
					arch=config.arch
				) for n_fts in config.kernel_features]

			###############################################################################################
			# Passing the generated word vectors to an LSTM network after transformations
			###############################################################################################

			# Two-layer highway network
			with tf.variable_scope("highway", reuse=tf.AUTO_REUSE, initializer=tf.initializers.random_uniform(-0.05, 0.05)):
				self.transformation_unit = TransformationUnit(config.num_dims)

			# LSTM network
			with tf.variable_scope("lstm", reuse=False, initializer=tf.initializers.random_uniform(-0.05, 0.05)):
				def create_lstm_cell():
					return tf.contrib.rnn.DropoutWrapper(
								tf.contrib.rnn.LSTMCell(
									num_units=config.num_units,
									cell_clip=config.lstm_clip,
									initializer=tf.initializers.random_uniform(-0.05, 0.05, seed=0),
									activation=tf.nn.relu,
									reuse=False
									),
								output_keep_prob = config.keep_prob)

				self.lstm_unit = tf.contrib.rnn.MultiRNNCell([create_lstm_cell() for _ in range(config.n_layers)], state_is_tuple=True)
							
				self.initial_states = tf.stack(self.lstm_unit.zero_state(self._batch_size, tf.float32))

			with tf.variable_scope("combined_unit", reuse=False, initializer=tf.initializers.random_uniform(-0.05, 0.05)):
				self.combined_unit = ConvPoolLSTMUnit(
					self.conv_units,
					self.pooling_windows,
					self.transformation_unit,
					self.lstm_unit,
					self.input_word_embedding,
					config.sliding_window_size
				)

				outputs, fstates = tf.nn.dynamic_rnn(
					self.combined_unit, 
					[char_vecs, inp_words], 
					initial_state=[tf.contrib.rnn.LSTMStateTuple(st[0], st[1]) for st in tf.unstack(self._states, axis=0)]
				)

				self.final_states = tf.stack([tf.stack([st.c, st.h]) for st in fstates])
				outputs = tf.reshape(outputs, [-1, self.combined_unit.output_size])

			# Projection and softmax layer
			output = tf.layers.dense(outputs, config.word_dims, name="projection")
			with tf.variable_scope("softmax", reuse=tf.AUTO_REUSE, initializer=tf.initializers.random_uniform(-0.05, 0.05)):
				logits = tf.matmul(output, self.output_word_embedding, transpose_b=True)
				logits = tf.reshape(logits, [-1, config.timesteps, config.word_vocab_size])

				self.prediction = tf.nn.softmax(logits, dim=2, name="prediction")

				self.loss = tf.reduce_sum(
					tf.nn.sparse_softmax_cross_entropy_with_logits(
						logits=logits, 
						labels=targets),
					axis=1) / config.timesteps

			# Optimizer
			with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE, initializer=tf.initializers.random_uniform(-0.05, 0.05)):
				# Standard tricks to train LSTMs
				tvars = tf.trainable_variables()
				grads = tf.gradients(self.loss, tvars)
				clipped_grads, _ = tf.clip_by_global_norm(grads, config.grad_clip)
				
				if config.optimizer == 'sgd':
					optim = tf.train.GradientDescentOptimizer(learning_rate=self._lr)
				elif config.optimizer == 'adam':
					optim = tf.train.AdamOptimizer(learning_rate=self._lr)
				elif config.optimizer == 'rmsprop':
					optim = tf.train.RMSPropOptimizer(learning_rate=self._lr, momentum=config.momentum)
				elif config.optimizer == 'momentum':
					optim = tf.train.MomentumOptimizer(learning_rate=self._lr, momentum=config.momentum)
				self.train_op = optim.apply_gradients(zip(clipped_grads, tvars), global_step=self.global_step)

			self.ce_loss_summary = tf.summary.scalar('cross_entropy_loss', tf.reduce_mean(self.loss))

			###############################################################################################
			# Fine tuning operations
			###############################################################################################
			
			self.fine_tune_op = dict()
			
			# find indices of cue words
			self.valid_cue_words = [i for i,j in enumerate(config.freq) 
										if int(j) > config.fine_tune_cue_threshold 
										and i not in range(config.word_vocab_size-4, config.word_vocab_size)]
			
			with tf.variable_scope("fine_tune", reuse=tf.AUTO_REUSE, initializer=tf.initializers.random_uniform(-0.05, 0.05)):
				# Normalized word embeddings
				ft_embeddings = self.input_word_embedding / (tf.norm(self.input_word_embedding, axis=1, keep_dims=True) + 1e-9)
				
				# ignore artificially added tokens
				ft_embeddings = ft_embeddings[:-4]
				
				# Placeholder for initial output embedding matrix (i.e. before fine-tuning)
				self.fine_tune_op['init_embed'] = tf.placeholder(
					tf.float32, [config.word_vocab_size, config.word_dims], name='initial_embedding')

				# Optimizer for fine-tuning
				ft_optim = tf.train.AdagradOptimizer(float(config.fine_tune_lr))

				# Accumulator for AP loss
				self.fine_tune_op['loss'] = 0.0

				# Find positive and negative word indices for given cue word
				# from the char-level input word embeddings
				ft_cosine_dist = tf.matmul(tf.gather(ft_embeddings, self.valid_cue_words), ft_embeddings, transpose_b=True)
				_, ft_pos_idx = tf.nn.top_k(ft_cosine_dist, config.fine_tune_pos_words)
				ft_neg_idx = tf.random_uniform(
					shape=(len(self.valid_cue_words), config.fine_tune_neg_words), 
					minval=0, 
					maxval=config.word_vocab_size-4, #ignore artificially added tokens
					dtype=tf.int32,
					seed=0)
				
				# Compute dot products of cue word with positive and negative words
				# Use output word embedding for obtaining word vectors
				cue_word_vecs = tf.gather(self.output_word_embedding, self.valid_cue_words)
				pos_vals = tf.einsum('ijk,ik->ij',
					tf.reshape(tf.gather(self.output_word_embedding, tf.reshape(ft_pos_idx, [-1])), 
								[len(self.valid_cue_words), config.fine_tune_pos_words, config.word_dims]),
					cue_word_vecs
				)
				neg_vals = tf.einsum('ijk,ik->ij',
					tf.reshape(tf.gather(self.output_word_embedding, tf.reshape(ft_neg_idx, [-1])), 
								[len(self.valid_cue_words), config.fine_tune_neg_words, config.word_dims]),
					cue_word_vecs
				)

				# Compute the Attract (A-) loss for every pair of pos-neg words
				for i in range(config.fine_tune_pos_words):
					for j in range(config.fine_tune_neg_words):
						self.fine_tune_op['loss'] = self.fine_tune_op['loss'] + tf.reduce_mean(tf.nn.relu(
							config.fine_tune_delta - pos_vals[:,i] + neg_vals[:,j]))

				# Compute the Preserve (-P) loss
				self.fine_tune_op['loss'] = self.fine_tune_op['loss'] + tf.reduce_mean(float(config.fine_tune_reg) * tf.norm(
					tf.gather(self.output_word_embedding, self.valid_cue_words) - 
					tf.gather(self.fine_tune_op['init_embed'], self.valid_cue_words) + 1e-9,
					axis=1))

				grads_vars = ft_optim.compute_gradients(
					loss=self.fine_tune_op['loss'],
					var_list=[self.output_word_embedding]
				)
				clipped_grads, _ = tf.clip_by_global_norm(
					t_list=[x[0] for x in grads_vars],
					clip_norm=float(config.fine_tune_grad_clip)
				)
				# Training op for fine-tuning
				self.fine_tune_op['tune'] = ft_optim.apply_gradients(
					[(clipped_grads[i], grads_vars[i][1]) for i in range(len(grads_vars))]
				)

			self.global_initializer = tf.global_variables_initializer()
			self.local_initializer = tf.local_variables_initializer()

			# Saver
			self.saver = tf.train.Saver(max_to_keep=3, var_list=tf.global_variables())

			self.ap_loss_summary = tf.summary.scalar('attract_preserve_loss', self.fine_tune_op['loss'])
			self.summary_writer = tf.summary.FileWriter(config.save_dir + '/logs/', tf.get_default_graph())

	def forward(self, sess, config, x, y=None, states=None, lr=1.0, mode='train'):
		""" Perform one forward and backward pass (only when required) over the network """
		
		# Storing the indices of input and output words
		word_idx = np.zeros([2, x.shape[0], x.shape[1]])
		word_lengths = np.zeros(x.shape)
		char_idx = np.zeros(x.shape + (config.max_word_length,))

		# Generate character indices, word lengths for obtaining word vectors
		# Also fill the input and output word indices in word_idx
		for i in range(x.shape[0]):
			for j in range(x.shape[1]):
				word_lengths[i,j] = len(x[i,j][2])
				char_idx[i, j, :len(x[i,j][2])] = x[i,j][2]
				word_idx[0, i, j] = x[i,j][0]
				if y is not None:
					word_idx[1, i, j] = y[i,j][0]
		
		# Execute ops depending on the mode
		if mode == 'train':
			res = sess.run([self.loss, self.final_states, self.train_op, self.ce_loss_summary], 
				feed_dict = {
					self._char_idx: char_idx,
					self._lengths: word_lengths,
					self._word_idx: word_idx,
					self._lr: lr,
					self._states: self.initial_states if states is None else states
				})
			self.summary_writer.add_summary(res[-1], self.global_step.eval(sess))
			return res[0], res[1] # ignore the output of assign_op
		elif mode == 'val' or mode == 'test':
			return sess.run(self.loss, feed_dict = {
				self._char_idx: char_idx,
				self._lengths: word_lengths,
				self._word_idx: word_idx,
				self._states: self.initial_states if states is None else states
			})
		elif mode == 'gen':
			return sess.run([self.prediction, self.final_states], feed_dict = {
				self._char_idx: char_idx,
				self._lengths: word_lengths,
				self._word_idx: word_idx,
				self._states: self.initial_states if states is None else states
			})

	def fine_tune(self, sess):
		""" Perform Attract-Preserve fine-tuning on embedding """
		# Store the original output embedding matrix
		org_output_embedding = np.squeeze(sess.run([self.output_word_embedding]))
		# Perform fine-tuning for fixed number of iterations
		for i in range(self.config.fine_tune_num_iters):
			st = time.time()
			total_loss, _, summ = sess.run(
				[self.fine_tune_op['loss'], self.fine_tune_op['tune'], self.ap_loss_summary], 
				feed_dict={self.fine_tune_op['init_embed']: org_output_embedding}
			)
			self.summary_writer.add_summary(summ, self.global_step.eval(sess) + i)
			print("Fine-tuning iteration: %d/%d, average AP loss: %.4f, time: %.2f" % 
					(i+1, self.config.fine_tune_num_iters, total_loss, time.time() - st))
			sys.stdout.flush()
