"""
charCNN-LSTM model that uses a mixture of word- and char-level embeddings.
"""

import tensorflow as tf
import numpy as np

class HybridEmbeddings(object):
    def __init__(self, config):
        """ Method for initializing model and constructing graph """
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Placeholders for input tensors/matrices
            self._idx = tf.placeholder(tf.int32, [None], name="input_indices")
            self._lengths = tf.placeholder(tf.int32, [config.batch_size * config.timesteps], name="input_lengths")
            self._output = tf.placeholder(tf.int32, [config.batch_size, config.timesteps], name="correct_output")
            self._states = tf.placeholder(tf.float32, [config.n_layers, 2, config.batch_size, config.num_units], name="lstm_states")
            self._lr = tf.placeholder_with_default(1.0, shape=[], name="learning_rate")
            
            # TensorFlow variables
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            self.best_metric = tf.Variable(1000, trainable=False, name="best_metric")

            # Other variables
            self.config = config

            # Create embeddings
            with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
                self.word_embedding = tf.get_variable("word", [config.word_vocab_size, config.word_dims])
                self.char_embedding = tf.get_variable("char", [config.char_vocab_size, config.char_dims])

            # Extract character vectors from embedding
            with tf.variable_scope("extract_char_vectors", reuse=tf.AUTO_REUSE):
                char_vecs = tf.gather(self.char_embedding, self._idx)
                char_vecs = tf.split(char_vecs, self._lengths)
                char_vecs = [tf.pad(cv, [[0, config.max_word_length-l], [0, 0]], 'CONSTANT', name="pad") 
                                for cv, l in zip(char_vecs, tf.unstack(self._lengths, axis=0))]
                char_vecs = tf.transpose(tf.stack(char_vecs), [0, 2, 1])
                char_vecs = tf.expand_dims(char_vecs, axis=3)

            # Apply convolution
            with tf.variable_scope("convolution", reuse=tf.AUTO_REUSE):
                feat_vecs = []
                for k, n_ft in zip(config.kernel_sizes, config.kernel_features):
                    fltr = tf.get_variable("filter_%d" % k, [config.char_vocab_size, k, 1, n_ft])
                    out_tensor = tf.nn.conv2d(char_vecs, fltr, strides=[1,1,1,1], padding="SAME")
                    # max-over time pooling
                    pooled_tensor = tf.nn.max_pool(out_tensor, 
                                                    ksize=[1, 1, config.max_word_length, 1],
                                                    strides=[1, 1, config.max_word_length, 1],
                                                    padding="VALID"
                                                    )
                    # squeeze the tensor to shape [batch_size*timesteps, n_fts]
                    squeezed_tensor = tf.squeeze(pooled_tensor)
                    feat_vecs.append(squeezed_tensor)
                # concatenate the features obtained
                feature_vec = tf.concat(feat_vecs, axis=2)
                self._input = tf.reshape(feature_vec, [config.batch_size, config.timesteps, config.num_dims])

            # Highway network
            with tf.variable_scope("highway", reuse=tf.AUTO_REUSE):
                x1 = tf.layers.dense(self._input, config.num_dims, activation=tf.nn.relu)
                t1 = tf.layers.dense(self._input, config.num_dims, activation=tf.nn.sigmoid)
                mod_input = x1*t1 + self._input*(1-t1)

            # LSTM network
            with tf.variable_scope("lstm", reuse=tf.AUTO_REUSE):
                self.lstm_cells = [tf.contrib.rnn.LSTMCell(
                                            num_units=config.num_units,
                                            cell_clip=config.lstm_clip,
                                            initializer=tf.random_uniform_initializer(-0.05, 0.05, seed=0),
                                            activation=tf.nn.relu,
                                            reuse=tf.AUTO_REUSE
                                            ) for _ in range(config.n_layers)]

                self.initial_states = []
                for i in range(config.n_layers):
                    st = self.lstm_cells[i].zero_state(config.batch_size, tf.float32)
                    self.initial_states.append(tf.stack([st.c, st.h], axis=0))

                self.initial_states = tf.stack(self.initial_states, axis=0, name="initial_states")

                inputs = mod_input
                states = tf.unstack(self._states, axis=0)
                states = [tf.contrib.rnn.LSTMStateTuple(st[0], st[1]) for st in states]
                self.final_states = []
                for i in range(config.n_layers):
                    with tf.variable_scope("layer_%d" % i, reuse=tf.AUTO_REUSE):
                        inputs, fstate = tf.nn.dynamic_rnn(self.lstm_cells[i], inputs, initial_state=states[i])
                        self.final_states.append(tf.stack([fstate.c, fstate.h], axis=0))

                self.final_states = tf.stack(self.final_states, axis=0, name="final_states")
                # reshape 'input', 'target' to [batch_size*timesteps, num_units]
                inputs = tf.reshape(inputs, [-1, config.num_units])
                targets = tf.reshape(self._output, [-1]) 

            # Projection and softmax layer
            with tf.variable_scope("projection", reuse=tf.AUTO_REUSE):
                w = tf.get_variable("p_w", [config.num_units, config.word_dims])
                b = tf.get_variable("p_b", config.word_dims)
                output = tf.matmul(inputs, w) + b
            with tf.variable_scope("softmax", reuse=tf.AUTO_REUSE):
                logits = tf.matmul(output, self.word_embedding, transpose_b=True)
                self.prediction = tf.nn.softmax(logits, axis=1, name="prediction")
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets, name="loss")

            # Optimizer
            with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
                # Standard tricks to train LSTMs
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), config.grad_clip)
                optim = tf.train.GradientDescentOptimizer(learning_rate=self._lr)
                self.train_op = optim.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            
            # Evaluation metric
            self.eval_metric = tf.reduce_mean(tf.exp(self.loss))    # trivial PPL

            # Saver
            self.saver = tf.train.Saver(max_to_keep=3)

    def forward(self, sess, x, y=None, states=None, lr=1.0, mode='train'):
        """ Perform one forward and backward pass (only when required) over the network """
        # Generate indices and lengths for obtaining word vectors

        print(states.shape, x.shape, y.shape)


        lengths = []; idx = []
        for b in x:
            for w in b:
                lengths += [len(w[2])]; idx += w[2]

        # Generate targets
        if y is not None:
            res = np.zeros(y.shape)
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    res[i,j] = y[i,j][0]
        
        # Execute ops depending on the mode
        if mode == 'train':
            return sess.run([self.loss, self.final_states, self.train_op], feed_dict = {
                self._idx: idx,
                self._lengths: lengths,
                self._output: res,
                self._lr: lr,
                self._states: self.initial_states if states is None else states,
            })
        elif mode == 'val':
            return sess.run(self.eval_metric, feed_dict = {
                self._idx: idx,
                self._lengths: lengths,
                self._output: res,
                self._states: self.initial_states if states is None else states,
            })
        elif mode == 'test':
            pass
         